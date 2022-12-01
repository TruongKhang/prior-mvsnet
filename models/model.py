import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from .module import depth_regression, conf_regression, CostRegNet, FeatureNet, RefineNet, get_depth_range_samples
from .utils.warping import homo_warping_3D, homo_warping_2D, masked_depth_conf
from .prior_net import PriorNet, UNet


Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self, in_channels):
        super(DepthNet, self).__init__()
        self.regress_cost = nn.ModuleList([nn.Conv3d(in_c, 1, 1) for in_c in in_channels])
        self.unet = nn.Sequential(UNet(5, 32, 1, 1, batchnorms=False),
                                  nn.Sigmoid(), nn.Threshold(0.05, 0.0))

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                log=False, est_vis=None, prior_info=None, stage_idx=0, img_idx=0):

        assert len(features) == proj_matrices.size(1), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = 0.0 #ref_volume
        #volume_sq_sum = ref_volume ** 2
        weight_sum = 0.0

        depth_info, mask_info = prior_info
        ref_depth, src_depths = depth_info
        ref_mask, src_masks = mask_info

        vis_maps = []
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            var = (ref_volume - warped_volume) ** 2 # [B, C, D, H, W]
            if stage_idx < len(self.regress_cost): #est_vis is None:
                rel_diff = (ref_depth - src_depths[:, src_idx, ...]).abs() / (ref_depth + 1e-10)

                simple_cost_vol = self.regress_cost[stage_idx](var).squeeze(1) #torch.sum(var, dim=1)
                max_cost, _ = torch.max(simple_cost_vol, dim=1, keepdim=True)
                # min_cost, _ = torch.min(simple_cost_vol, dim=1, keepdim=True)
                mean_cost = torch.mean(simple_cost_vol, dim=1, keepdim=True)
                feats = torch.cat((max_cost, mean_cost, rel_diff, ref_mask.float(), src_masks[:, src_idx, ...].float()), dim=1)
                vis_map = self.unet(feats)
                if stage_idx == (len(self.regress_cost) - 1):
                    vis_map_png = (vis_map.cpu().numpy() * 1000).astype(np.uint16)
                    F_occ = (rel_diff.cpu().numpy() * 10000).astype(np.uint16)
                    F_occ_conf = src_masks[:, src_idx, ...] #torch.min(ref_mask, src_masks[:, src_idx, ...])
                    F_occ_conf = (F_occ_conf.cpu().numpy() * 100).astype(np.uint16)
                    if not os.path.exists("vis_features/%d" % img_idx):
                        os.makedirs("vis_features/%d" % img_idx)
                    #plt.imsave("vis_features/%d/vis_map_color%d.png" % (img_idx, src_idx), vis_map_png[0][0])
                    plt.imsave("vis_features/%d/f_occ_color%d.png" % (img_idx, src_idx), F_occ[0][0])
                    plt.imsave("vis_features/%d/f_occ_conf_color%d.png" % (img_idx, src_idx), F_occ_conf[0][0])
                    #vis_map_png = Image.fromarray(vis_map_png[0][0])
                    #vis_map_png.save("vis_features/%d/vis_map%d.png" % (img_idx, src_idx))
                    F_occ_png = Image.fromarray(F_occ[0][0])
                    F_occ_png.save("vis_features/%d/f_occ%d.png" % (img_idx, src_idx))
                    F_occ_conf_png = Image.fromarray(F_occ_conf[0][0])
                    F_occ_conf_png.save("vis_features/%d/f_occ_conf%d.png" % (img_idx, src_idx))
            else:
                vis_map = est_vis[:, src_idx, ...]
            vis_maps.append(vis_map.detach())
            vis_map = vis_map.unsqueeze(2)
            volume_sum = volume_sum + var * vis_map
            weight_sum = weight_sum + vis_map
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sum / (weight_sum + 1e-10) #(num_views - 1)

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        # log_prob_volume = F.log_softmax(prob_volume_pre, dim=1)
        prob_volume = F.softmax(prob_volume_pre, dim=1) if not log else F.log_softmax(prob_volume_pre, dim=1)

        return prob_volume, torch.stack(vis_maps, dim=1)


class SeqProbMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8), num_stages=3, pretrained_prior=None,
                 pretrained_mvs=None, pretrained_model=None):
        super(SeqProbMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = num_stages #len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(3)])
        if self.refine:
            self.refine_network = RefineNet(self.feature.out_channels[-1] + 2)

        if self.num_stage == 3:
            self.dnet = DepthNet(self.feature.out_channels[:2])
        else:
            self.dnet = DepthNet(self.feature.out_channels[:3])

        self.pr_net = PriorNet()

        if pretrained_model is not None:
            print("Loading pretrained prior-mvsnet model")
            ckpt = torch.load(pretrained_model)
            new_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
            self.load_state_dict(new_state_dict, strict=True)

        if pretrained_prior is not None:
            print("Loading pretrained PriorNet")
            ckpt = torch.load(pretrained_prior)
            new_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
            self.pr_net.load_state_dict(new_state_dict)
            for p in self.pr_net.parameters():
                p.requires_grad = False

        if pretrained_mvs is not None:
            print("Loading pretrained MVS-Net")
            ckpt = torch.load(pretrained_mvs)
            feat_dict, cost_reg_dict = {}, {}
            for k, v in ckpt['model'].items():
                if "feature" in k:
                    feat_dict[k.replace("feature.", "")] = v
                if "cost_regularization" in k:
                    cost_reg_dict[k.replace("cost_regularization.", "")] = v
            self.feature.load_state_dict(feat_dict)
            self.cost_regularization.load_state_dict(cost_reg_dict)

        self.mvsnet_parameters = list(self.feature.parameters()) + list(self.cost_regularization.parameters()) + list(self.dnet.parameters()) #list(self.vis_net.parameters())

    def get_log_prior(self, depth, conf, depth_values):
        min_depth, max_depth = depth_values[:, [0], ...], depth_values[:, [-1], ...]
        mask = (depth.detach() >= min_depth) & (depth.detach() <= max_depth)
        mask = mask.repeat(1, depth_values.size(1), 1, 1)
        log_dist_masked = torch.zeros_like(depth_values)
        log_dist = - conf * torch.abs(depth - depth_values) + torch.log(conf + 1e-16)
        log_dist_masked[mask] = log_dist[mask]
        log_dist_masked = F.log_softmax(log_dist_masked, dim=1)
        return log_dist_masked

    def forward(self, imgs, proj_matrices, depth_values, prior=None, depth_scale=1.0, src_prior=None, gt_vis=None, img_idx=0):

        depth_min = depth_values[:, [0]].unsqueeze(-1).unsqueeze(-1)
        depth_max = depth_values[:, [-1]].unsqueeze(-1).unsqueeze(-1)
        depth_interval = (depth_max - depth_min).squeeze(1) / depth_values.size(1) #(depth_values[:, 1] - depth_values[:, 0]).unsqueeze(-1).unsqueeze(-1)

        # depth_min = float(depth_values[0, 0].cpu().numpy())
        # depth_max = float(depth_values[0, -1].cpu().numpy())
        # depth_interval = (depth_max - depth_min) / depth_values.size(1)
        # dmap_min, dmap_max = depth_values[:, 0], depth_values[:, -1]

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        batch_size, height, width = imgs.size(0), imgs.size(3), imgs.size(4)
        if self.num_stage == 4:
            height, width = height // 2, width // 2

        outputs = {}
        depth, cur_depth = None, None

        vis_maps = None
        for stage_idx in range(3):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            stage_name = "stage{}".format(stage_idx + 1)
            features_stage = [feat[stage_name] for feat in features]
            proj_matrices_stage = proj_matrices[stage_name]
            stage_scale = self.stage_infos[stage_name]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height, width], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            if vis_maps is not None:
                vis_maps = F.interpolate(vis_maps.view(-1, 1, vis_maps.size(3), vis_maps.size(4)),
                                         [height // int(stage_scale), width // int(stage_scale)], mode='nearest')
                vis_maps = vis_maps.view(batch_size, -1, 1, height//int(stage_scale), width//int(stage_scale))

            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=imgs[0].dtype,
                                                        device=imgs[0].device, min_depth=depth_min, max_depth=depth_max,
                                                        shape=[batch_size, height, width])

            # added by Khang
            depth_values_stage = F.interpolate(depth_range_samples.unsqueeze(1),
                                               [self.ndepths[stage_idx], height//int(stage_scale), width//int(stage_scale)],
                                               mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)

            if prior is not None:
                prior_depths, prior_confs = prior[stage_name]
                prior_masks = prior_depths > 0
                # normalize depth in to DTU range
                prior_depths = (prior_depths - depth_min.unsqueeze(1)) / depth_scale.unsqueeze(1) + 425
                prior_depths[prior_depths < 425] = 0.0
                scaling_depth_values_stage = (depth_values_stage - depth_min) / depth_scale + 425

                est_prior_depth, est_prior_conf, prop_prior_depths, prop_prior_confs = self.pr_net(prior_depths, prior_confs)
                log_prior = self.get_log_prior(est_prior_depth, est_prior_conf, scaling_depth_values_stage) # / depth_scale)
            else:
                log_prior = 0.0
                est_prior_depth, est_prior_conf = None, None

            all_prior_depths = est_prior_depth.detach(), prop_prior_depths #prior[stage_name][0]
            all_prior_masks = est_prior_conf.detach(), prop_prior_confs #(prior[stage_name][0] > 0)
            log_likelihood, vis_maps = self.dnet(features_stage, proj_matrices_stage, depth_values=depth_values_stage,
                                       num_depth=self.ndepths[stage_idx],
                                       cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                       log=True, est_vis=vis_maps, prior_info=(all_prior_depths, all_prior_masks), stage_idx=stage_idx, img_idx=img_idx)
 
            log_posterior = log_likelihood + log_prior
            posterior_vol = F.softmax(log_posterior, dim=1)
            depth = depth_regression(posterior_vol, depth_values_stage)
            final_conf = conf_regression(posterior_vol)
            var = torch.abs(depth.unsqueeze(1).detach() - depth_values_stage)
            var = depth_regression(posterior_vol, var)

            outputs_stage = {"depth": depth, "photometric_confidence": final_conf, "var": var,
                             "prior_depth": est_prior_depth, "prior_conf": est_prior_conf, "vis_maps": vis_maps}
            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)

        if self.num_stage == 4:
            height, width = height * 2, width * 2
        total_dist = 0.0
        final_depth = F.interpolate(depth.detach().unsqueeze(1), [height, width], mode='nearest')
        for stage_idx in range(3):
            stage_name = "stage{}".format(stage_idx + 1)
            depth_stage, var_stage = outputs[stage_name]["depth"].detach(), outputs[stage_name]["var"].detach()
            depth_stage = F.interpolate(depth_stage.unsqueeze(1), [height, width], mode='nearest')
            var_stage = F.interpolate(var_stage.unsqueeze(1), [height, width], mode='nearest')
            dist = torch.exp(- (depth_stage - final_depth).abs() / (var_stage + depth_scale)) / (var_stage / depth_scale + 1.0)
            total_dist = total_dist + dist
        total_dist /= 3
        final_conf = total_dist
        feat_img = features[0]["stage%d" % self.num_stage]
        # depth map refinement
        if self.refine:
            final_depth = (final_depth - depth_min) / depth_scale + 425
            final_depth, final_conf = self.refine_network(final_depth, final_conf, feat_img)
            final_depth = (final_depth - 425) * depth_scale.squeeze(1) + depth_min.squeeze(1)
        else:
            final_depth = final_depth.squeeze(1)
            final_conf = final_conf.squeeze(1)
        outputs["final_depth"] = final_depth #* depth_scale
        outputs["final_conf"] = final_conf

        return outputs
