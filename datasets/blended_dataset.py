from torch.utils.data import Dataset
import os, cv2, time
from PIL import Image, ImageOps
from datasets.data_io import *


class BlendedMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(BlendedMVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        if self.mode == 'test':
            scans = self.listfile
        else:
            with open(self.listfile) as f:
                scans = f.readlines()
                scans = [line.rstrip() for line in scans]

        interval_scale_dict = {}
        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        # src_views = src_views[:(self.nviews-1)]
                        metas.append((scan, ref_view, src_views, scan))

        # self.interval_scale = interval_scale_dict
        print("dataset ", self.mode, "metas: ", len(metas), "interval_scale: {}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        #intrinsics[0, 2] -= 64.0
        #intrinsics[1, 2] -= 32.0
        # intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def prepare_img(self, img):
        h, w = img.shape[:2]
        target_h, target_w = 576, 768 #512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        img_crop = img[start_h: start_h + target_h, start_w: start_w + target_w]
        return img_crop

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = self.prepare_img(np_img)

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = self.prepare_img(depth)
        h, w = depth.shape
        depth_ms = {}
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            depth_ms["stage%d" % (i + 1)] = cv2.resize(depth, (w // (2 ** p), h // (2 ** p)),
                                                       interpolation=cv2.INTER_NEAREST)

        return depth_ms

    def read_mask(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        np_img = (depth > 0).astype(np.float32)
        np_img = self.prepare_img(np_img)
        # np_img = cv2.resize(np_img, (768, 576), interpolation=cv2.INTER_NEAREST)
        h, w = np_img.shape
        np_img_ms = {}
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            np_img_ms["stage%d" % (i + 1)] = cv2.resize(np_img, (w // (2 ** p), h // (2 ** p)),
                                                        interpolation=cv2.INTER_NEAREST)
        return np_img_ms

    def read_prior(self, depth_file, conf_file, filetype='png'):
        depth_file = '{}.{}'.format(depth_file, filetype)
        conf_file = '{}.{}'.format(conf_file, filetype)
        if filetype == 'png':
            prior_depth = np.array(Image.open(depth_file), dtype=np.float32) / 1000
            prior_conf = np.array(Image.open(conf_file), dtype=np.float32) / 255
        else:
            prior_depth = np.array(read_pfm(depth_file)[0], dtype=np.float32)
            prior_conf = np.array(read_pfm(conf_file)[0], dtype=np.float32)
        return prior_depth, prior_conf

    def __getitem__(self, idx):
        # global s_h, s_w
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        # if self.mode == 'train':
        #     src_views = src_views[:7]
        #     np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews-1)]

        imgs = []
        depth_values = None
        proj_matrices = []
        mask, depth_ms = None, None
        input_depths = {"stage1": [], "stage2": [], "stage3": []}
        input_confs = {"stage1": [], "stage2": [], "stage3": []}
        input_masks = {"stage1": [], "stage2": [], "stage3": []}
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask(mask_filename)
                depth_ms = self.read_depth(depth_filename)

                # get depth values
                depth_max = depth_interval * (self.ndepths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

            if self.kwargs['load_prior'] is not None:
                mask_vid = self.read_mask(mask_filename)
                for stage in input_masks.keys():
                    input_masks[stage].append(mask_vid[stage])

                prior_depth_file = os.path.join(self.datapath, 'priors/{}/depth_est/{:0>8}'.format(scan, vid))
                prior_conf_file = os.path.join(self.datapath, 'priors/{}/confidence/{:0>8}'.format(scan, vid))
                p_depth, p_conf = self.read_prior(prior_depth_file, prior_conf_file, filetype='png')
                height, width = p_depth.shape
                for i in range(3):
                    p = self.kwargs["num_stages"] - i - 1
                    input_depths["stage%d" % (i + 1)].append(
                        cv2.resize(p_depth, (width // (2 ** p), height // (2 ** p)),
                                   interpolation=cv2.INTER_NEAREST))
                    input_confs["stage%d" % (i + 1)].append(cv2.resize(p_conf, (width // (2 ** p), height // (2 ** p)),
                                                                       interpolation=cv2.INTER_NEAREST))

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices_ms = {}
        proj_matrices = np.stack(proj_matrices)
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            stage_projmats = proj_matrices.copy()
            stage_projmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / (2 ** p)
            proj_matrices_ms["stage%d" % (i + 1)] = stage_projmats

        outputs = {}
        if self.kwargs['load_prior'] is not None:
            for stage in input_depths.keys():
                input_depths[stage] = np.expand_dims(np.stack(input_depths[stage]), axis=1)
                input_confs[stage] = np.expand_dims(np.stack(input_confs[stage]), axis=1)
                input_masks[stage] = np.expand_dims(np.stack(input_masks[stage]), axis=1)
            outputs.update({"prior_depths": input_depths, "prior_confs": input_confs, "prior_masks": input_masks})

        return outputs.update({"imgs": imgs, "proj_matrices": proj_matrices_ms,
                               "depth": depth_ms, "depth_values": depth_values, "mask": mask,
                               "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"})