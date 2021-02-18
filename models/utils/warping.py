import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import MYTH


def parse_intrinsics(intrinsics):
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z, device=intrinsics.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, ndepths = depth.size(0), depth.size(1)
    # height, width = img_shape
    # y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
    #                        torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    # y, x = y.contiguous(), x.contiguous()
    # y, x = y.view(height * width), x.view(height * width)

    x_cam = xy[..., 0].view(batch_size, -1) # ndepths, -1)
    y_cam = xy[..., 1].view(batch_size, -1) # ndepths, -1)
    z_cam = depth.view(batch_size, -1) # ndepths, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1) #1, 3, 2)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)
    # world_coords = torch.matmul(cam2world.unsqueeze(1), pixel_points_cam).permute(0, 1, 3, 2)[..., :3]

    return world_coords


def homo_warping_3D(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    #with torch.no_grad():
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
    grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros') #, align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping_2D(depths, cfds, projs, ref_proj=None):
    if ref_proj is not None:
        projs = torch.cat((ref_proj, projs), 1)
        fake_depth = torch.zeros_like(depths[:, [0], ...])
        fake_conf = torch.zeros_like(cfds[:, [0], ...])
        depths = torch.cat((fake_depth, depths), dim=1)
        cfds = torch.cat((fake_conf, cfds), dim=1)

    intrinsics, extrinsics = projs[:, :, 1, :, :], projs[:, :, 0, :, :]
    projs = torch.matmul(intrinsics[..., :3, :3], extrinsics[..., :3, :4])

    warped_depths, warped_cfds, _ = MYTH.DepthColorAngleReprojectionNeighbours.apply(depths, cfds, projs, 1.0)
    warped_depths = warped_depths[:, 1:, ...]
    warped_cfds = warped_cfds[:, 1:, ...]
    return warped_depths, warped_cfds


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    device = depth_ref.device
    batch_size, width, height = depth_ref.size(0), depth_ref.size(2), depth_ref.size(1)
    # step1. project reference pixels to the source view
    # ref, src projection matrices
    ref_proj = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    src_proj = ref_proj.clone()
    ref_proj[:, :3, :] = torch.matmul(intrinsics_ref, extrinsics_ref) # [B, 4, 4]
    src_proj[:, :3, :] = torch.matmul(intrinsics_src, extrinsics_src) # [B, 4, 4]
    ref2src_proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # [B, 4, 4]

    # reference view x, y
    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                                   torch.arange(0, width, dtype=torch.float32, device=device)])
    x_ref, y_ref = x_ref.contiguous().view(-1), y_ref.contiguous().view(-1)
    # reference 3D space
    xyz_ref = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))) # [3, H*W]
    xyz_ref = xyz_ref.unsqueeze(0).repeat(batch_size, 1, 1) * depth_ref.view(batch_size, 1, -1) # [B, 3, H*W]
    xyz_ref = torch.cat((xyz_ref, torch.ones((batch_size, 1, height*width), dtype=torch.float32, device=device)), dim=1) # [B, 4, H*W]
    # source 3D space
    proj_xyz = torch.matmul(ref2src_proj, xyz_ref)[:, :3, :] # [B, 3, H*W]
    # source view x, y
    xy_src = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]
    x_src, y_src = xy_src[:, 0, :].view(batch_size, height, width), xy_src[:, 1, :].view(batch_size, height, width)
    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_normalized = x_src / ((width - 1) / 2) - 1
    y_normalized = y_src / ((height - 1) / 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=3) # [B, H, W, 2]
    sampled_depth_src = F.grid_sample(depth_src.unsqueeze(1), grid) # [B, 1, H, W]

    xyz_src = torch.cat((xy_src, torch.ones((batch_size, 1, height*width), device=device, dtype=torch.float32)), dim=1)
    xyz_src = xyz_src * sampled_depth_src.view(batch_size, 1, -1) # [B, 3, H*W]
    xyz_src = torch.cat((xyz_src, torch.ones((batch_size, 1, height*width), dtype=torch.float32, device=device)), dim=1) # [B, 4, H*W]
    src2ref_proj = torch.matmul(ref_proj, torch.inverse(src_proj)) # [B, 4, 4]
    xyz_reprojected = torch.matmul(src2ref_proj, xyz_src)[:, :3, :] # [B, 3, H*W]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2, :].view(batch_size, height, width)
    xy_reprojected = xyz_reprojected[:, :2, :] / xyz_reprojected[:, 2:3, :]
    x_reprojected = xy_reprojected[:, 0, :].view(batch_size, height, width)
    y_reprojected = xy_reprojected[:, 1, :].view(batch_size, height, width)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    device = depth_ref.device
    batch_size, width, height = depth_ref.size(0), depth_ref.size(2), depth_ref.size(1)
    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                                   torch.arange(0, width, dtype=torch.float32, device=device)])
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref.unsqueeze(0)) ** 2 + (y2d_reprojected - y_ref.unsqueeze(0)) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / (depth_ref + 1e-16)
    mask = (dist < 2) & (relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def masked_depth_conf(depths, confs, proj_matrices, thres_view=3, thres_conf=0.9):
    nviews = depths.size(1)
    intrinsics, extrinsics = proj_matrices[:, :, 1, :3, :3], proj_matrices[:, :, 0, :3, :4]
    # for each reference view and the corresponding source views
    all_masks = []
    for id_ref in range(nviews):
        depth_ref, conf_ref = depths[:, id_ref, ...], confs[:, id_ref, ...]
        intrinsics_ref, extrinsics_ref = intrinsics[:, id_ref, ...], extrinsics[:, id_ref, ...]
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        # compute the geometric mask
        geo_mask_sum = 0
        photo_mask = conf_ref > thres_conf
        for id_src in range(nviews):
            if id_src != id_ref:
                depth_src, conf_src = depths[:, id_src, ...], confs[:, id_src, ...]
                intrinsics_src, extrinsics_src = intrinsics[:, id_src, ...], extrinsics[:, id_src, ...]
                geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(depth_ref, intrinsics_ref,
                                                                                            extrinsics_ref, depth_src,
                                                                                            intrinsics_src, extrinsics_src)
                geo_mask_sum += geo_mask.float()
                all_srcview_depth_ests.append(depth_reprojected)
                all_srcview_x.append(x2d_src)
                all_srcview_y.append(y2d_src)
                all_srcview_geomask.append(geo_mask)

        # at least 3 source views matched
        geo_mask = geo_mask_sum >= thres_view
        final_mask = photo_mask & geo_mask
        all_masks.append(final_mask)
    all_masks = torch.stack(all_masks, dim=1).float()
    return depths * all_masks, confs * all_masks


def get_prior(depths, confs, project_matrices, depth_scale=1.0):
    prior = {}
    src_depths, src_confs = depths[:, 1:, 0, ...], confs[:, 1:, 0, ...]
    src_proj_matrices = project_matrices[:, 1:, ...]
    filtered_src_depths, filtered_src_confs = masked_depth_conf(src_depths, src_confs, src_proj_matrices, thres_view=1,
                                                                thres_conf=0.1)
    depths[:, 1:, 0, ...] = filtered_src_depths
    confs[:, 1:, 0, ...] = (filtered_src_confs > 0).float()
    warped_depths, warped_confs = homo_warping_2D(depths, confs, project_matrices)
    warped_depths /= depth_scale

    scale = {"stage1": 4, "stage2": 2, "stage3": 1}
    H, W = warped_depths.size(3), warped_depths.size(4)
    for stage in project_matrices.keys():
        warped_depths_stage = F.interpolate(warped_depths.squeeze(2), [H // scale[stage], W // scale[stage]],
                                            mode='nearest')
        warped_confs_stage = F.interpolate(warped_confs.squeeze(2), [H // scale[stage], W // scale[stage]],
                                           mode='nearest')
        prior[stage] = warped_depths_stage.unsqueeze(2), warped_confs_stage.unsqueeze(2)
    return prior


if __name__ == '__main__':
    from datasets.data_loaders import DTULoader
    from tensorboardX import SummaryWriter
    from utils import tensor2numpy, save_images, tocuda

    writer = SummaryWriter(".")
    device = torch.device('cuda:0')
    datapath = "/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/train"
    stage = "stage3"
    data_loader = DTULoader(datapath, '/home/khangtg/Documents/lab/seq-prob-mvs/lists/dtu/subsub_train.txt', 'train', 5, 192, 1.06, shuffle=False)
    for batch_idx, sample in enumerate(data_loader):
        print(batch_idx)
        sample_cuda = tocuda(sample)
        proj_matrices = sample_cuda["proj_matrices"]["stage3"]
        depths = sample_cuda["prior_depths"]["stage3"].squeeze(2)
        confs = sample_cuda["prior_confs"]["stage3"].squeeze(2)
        filtered_depths, filtered_confs = filter_depth(depths, confs, proj_matrices, thres_view=4, thres_conf=0.9)

        nviews = filtered_depths.size(1)
        dict_views = {}
        for i in range(nviews):
            dict_views["filtered_depth_view_%d" % (i + 1)] = filtered_depths[:, i, ...]
            dict_views["depth_view_%d" % (i + 1)] = depths[:, i, ...]
        save_images(writer, 'train', tensor2numpy(dict_views), batch_idx)
        if batch_idx == 5:
            break

