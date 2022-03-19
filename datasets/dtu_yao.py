from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import read_pfm
import random

np.random.seed(123)
random.seed(123)


# the DTU dataset preprocessed by Yao Yao (only for training)
class DTUDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(DTUDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    src_views = src_views[:(self.nviews-1)]

                    # f.readline() # ignore the given source views
                    # src_views = [x for x in range(left, left+self.nviews) if x != ref_view]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
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
        # rescale intrinsics to full image-resolution
        intrinsics[:2, :] *= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {}
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            np_img_ms["stage%d" % (i + 1)] = cv2.resize(np_img, (w // (2 ** p), h // (2 ** p)),
                                                        interpolation=cv2.INTER_NEAREST)
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {}
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            depth_lr_ms["stage%d" % (i + 1)] = cv2.resize(depth_lr, (w // (2 ** p), h // (2 ** p)),
                                                          interpolation=cv2.INTER_NEAREST)
        return depth_lr_ms

    def read_prior(self, depth_file, conf_file, filetype='png'):
        depth_file = '{}.{}'.format(depth_file, filetype)
        conf_file = '{}.{}'.format(conf_file, filetype)
        if filetype == 'png':
            prior_depth = np.array(Image.open(depth_file), dtype=np.float32) / 10
            prior_conf = np.array(Image.open(conf_file), dtype=np.float32) / 255
        else:
            prior_depth = np.array(read_pfm(depth_file)[0], dtype=np.float32)
            prior_conf = np.array(read_pfm(conf_file)[0], dtype=np.float32)
        return prior_depth, prior_conf

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views

        view_ids = [ref_view] + src_views

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []
        input_depths = {"stage1": [], "stage2": [], "stage3": []}
        input_confs = {"stage1": [], "stage2": [], "stage3": []}
        input_masks = {"stage1": [], "stage2": [], "stage3": []}

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0: # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                #get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

            if self.kwargs['load_prior']:
                mask_vid = self.read_mask_hr(mask_filename_hr)
                for stage in input_masks.keys():
                    input_masks[stage].append(mask_vid[stage])

                stage = "stage3"
                prior_depth_file = os.path.join(self.datapath,
                                                'priors/{}/{}/depth_est/{:0>3}_{}'.format(stage, scan, vid, light_idx))
                prior_conf_file = os.path.join(self.datapath,
                                               'priors/{}/{}/confidence/{:0>3}_{}'.format(stage, scan, vid, light_idx))
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
        #ms proj_mats
        proj_matrices_ms = {}
        proj_matrices = np.stack(proj_matrices)
        for i in range(self.kwargs["num_stages"]):
            p = self.kwargs["num_stages"] - i - 1
            stage_projmats = proj_matrices.copy()
            stage_projmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / (2 ** p)
            proj_matrices_ms["stage%d" % (i + 1)] = stage_projmats

        outputs = {}
        if self.kwargs['load_prior']:
            for stage in input_depths.keys():
                input_depths[stage] = np.expand_dims(np.stack(input_depths[stage]), axis=1)
                input_confs[stage] = np.expand_dims(np.stack(input_confs[stage]), axis=1)
                input_masks[stage] = np.expand_dims(np.stack(input_masks[stage]), axis=1)
            outputs.update({"prior_depths": input_depths, "prior_confs": input_confs, "prior_masks": input_masks})

        outputs.update({"imgs": imgs,
                        "proj_matrices": proj_matrices_ms,
                        "depth": depth_ms,
                        "depth_values": depth_values,
                        "mask": mask})
        return outputs
