import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)

import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

import torch
from bisect import bisect_right
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d

def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u] #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))


# class PatchOp:
#     '''
#     A op class to extract patches from an image, and overlap-average patches
#     back to an image, with pre-defined image size, patch size and stride.
#     '''
#
#     def __init__(self, bsz, height, width, psz, stride):
#         self.psz, self.stride = psz, stride
#         self.extract_op = tf.extract_image_patches(
#             tf.ones([bsz, height, width, 1]), [1, psz, psz, 1],
#             [1, stride, stride, 1], [1, 1, 1, 1], 'VALID').op
#         self.norm = _ExtractImagePatchesGrad(
#             self.extract_op, self.extract_op.outputs[0])[0]
#
#     def extract_patches(self, image):
#         patches = tf.extract_image_patches(
#             image, [1, self.psz, self.psz, 1],
#             [1, self.stride, self.stride, 1], [1, 1, 1, 1], 'VALID')
#         return patches
#
#     def group_patches(self, patches):
#         image = _ExtractImagePatchesGrad(self.extract_op, patches)[0]
#         return image / self.norm
#
#     def group_extract(self, patches):
#         ''' first group, then extract'''
#         image = self.group_patches(patches)
#         out = self.extract_patches(image)
#         return image, out
#
#
# # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/array_grad.py#L725
#
# from math import ceil
#
# from tensorflow.python.framework import sparse_tensor
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import sparse_ops
#
# def _ExtractImagePatchesGrad(op, grad):
#     ''' Gradient function of tf.extract_image_patches. '''
#
#     batch_size, rows_in, cols_in, channels = [
#         dim.value for dim in op.inputs[0].get_shape()]
#     input_bhwc = array_ops.shape(op.inputs[0])
#     batch_size = input_bhwc[0]
#     channels = input_bhwc[3]
#
#     _, rows_out, cols_out, _ = [dim.value for dim in op.outputs[0].get_shape()]
#     _, ksize_r, ksize_c, _ = op.get_attr("ksizes")
#     _, stride_r, stride_h, _ = op.get_attr("strides")
#     _, rate_r, rate_c, _ = op.get_attr("rates")
#     padding = op.get_attr("padding")
#
#     ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
#     ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)
#
#     if padding == b"SAME":
#         rows_out = int(ceil(rows_in / stride_r))
#         cols_out = int(ceil(cols_in / stride_h))
#         pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
#         pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2
#
#     elif padding == b"VALID":
#         rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
#         cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
#         pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
#         pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in
#
#     pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)
#
#     grad_expanded = array_ops.transpose(
#         array_ops.reshape(
#             grad, (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)),
#         (1, 2, 3, 4, 0, 5))
#     grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))
#
#     row_steps = range(0, rows_out * stride_r, stride_r)
#     col_steps = range(0, cols_out * stride_h, stride_h)
#
#     idx = []
#     for i in range(rows_out):
#         for j in range(cols_out):
#             r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
#             r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff
#
#             idx.extend([(r * (cols_in) + c, i * (cols_out * ksize_r * ksize_c) + j
#                          * (ksize_r * ksize_c) + ri * (ksize_c) + ci)
#                         for (ri, r) in enumerate(range(r_low, r_high, rate_r))
#                         for (ci, c) in enumerate(range(c_low, c_high, rate_c))
#                         if 0 <= r and r < rows_in and 0 <= c and c < cols_in])
#
#     sp_shape = (rows_in * cols_in, rows_out * cols_out * ksize_r * ksize_c)
#
#     sp_mat = sparse_tensor.SparseTensor(
#         array_ops.constant(idx, dtype=ops.dtypes.int64),
#         array_ops.ones((len(idx),), dtype=ops.dtypes.float32), sp_shape)
#
#     jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)
#
#     grad_out = array_ops.reshape(jac, (rows_in, cols_in, batch_size, channels))
#     grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
#
#     return [grad_out]