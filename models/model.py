import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import depth_regression, conf_regression, CostRegNet, FeatureNet, RefineNet, get_depth_range_samples
from .utils.warping import homo_warping_3D
from .prior_net import PriorNet
from .visibility import SharedMLP


Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self, feature_channels=(8, 8, 8), depth_channels=0,
                 occ_shared_channels=(128, 128, 128), occ_global_channels=(64, 16, 4)):
        super(DepthNet, self).__init__()
        self.occ_shared_mlp = nn.ModuleList([SharedMLP(2 * c + depth_channels, occ_shared_channels, ndim=2, bn=True)
                                             for c in feature_channels])
        self.occ_global_mlp = SharedMLP(occ_shared_channels[-1], occ_global_channels, bn=False)
        self.occ_pred = nn.Sequential(nn.Conv1d(occ_global_channels[-1], 1, 1), nn.Sigmoid())

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                log=False, stage_idx=0, depth_min=None, depth_max=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # normalize depth
        batch_size, num_channels, height, width = features[0].size()
        depth_min, depth_max = depth_min.view(-1, 1, 1, 1), depth_max.view(-1, 1, 1, 1)
        norm_depth_values = (depth_values - depth_min) / (depth_max - depth_min)
        norm_depth_values = norm_depth_values.unsqueeze(1)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        weight_sum = 1.0
        # del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            vis_inputs = torch.cat((ref_volume, warped_volume, norm_depth_values), dim=1)
            vis_inputs = vis_inputs.view(batch_size, 2*num_channels+1, num_depth, -1)
            vis_map = self.occ_shared_mlp[stage_idx](vis_inputs)
            vis_map, _ = torch.max(vis_map, dim=2)
            vis_map = self.occ_pred(self.occ_global_mlp(vis_map)) # [B, 1, H*W]
            vis_map = vis_map.view(batch_size, 1, 1, height, width)
            weight_sum = weight_sum + vis_map

            if self.training:
                volume_sum = volume_sum + warped_volume * vis_map
                volume_sq_sum = volume_sq_sum + (warped_volume ** 2) * vis_map
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume * vis_map
                volume_sq_sum += warped_volume.pow_(2) * vis_map  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        # volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        volume_variance = volume_sq_sum.div_(weight_sum).sub_(volume_sum.div_(weight_sum).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        # log_prob_volume = F.log_softmax(prob_volume_pre, dim=1)
        prob_volume = F.softmax(prob_volume_pre, dim=1) if not log else F.log_softmax(prob_volume_pre, dim=1)

        return prob_volume


class SeqProbMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8), pretrained_prior=None, pretrained_mvs=None, use_prior=True):
        super(SeqProbMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
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
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet(self.feature.out_channels[-1] + 2)

        self.dnet = DepthNet(feature_channels=self.feature.out_channels, depth_channels=1,
                             occ_shared_channels=(32,), occ_global_channels=(4,))
        if use_prior:
            self.pr_net = PriorNet()
            if pretrained_prior is not None:
                ckpt = torch.load(pretrained_prior)
                self.pr_net.load_state_dict(ckpt['state_dict'])
                for p in self.pr_net.parameters():
                    p.requires_grad = False

        if pretrained_mvs is not None:
            ckpt = torch.load(pretrained_mvs)
            feat_dict, cost_reg_dict = {}, {}
            for k, v in ckpt['model'].items():
                if "feature" in k:
                    feat_dict[k.replace("feature.", "")] = v
                if "cost_regularization" in k:
                    cost_reg_dict[k.replace("cost_regularization.", "")] = v
            self.feature.load_state_dict(feat_dict)
            self.cost_regularization.load_state_dict(cost_reg_dict)

        self.mvsnet_parameters = list(self.feature.parameters()) + list(self.cost_regularization.parameters()) + list(self.dnet.parameters())

    def get_log_prior(self, depth, conf, depth_values):
        min_depth, max_depth = depth_values[:, [0], ...], depth_values[:, [-1], ...]
        mask = (depth.detach() >= min_depth) & (depth.detach() <= max_depth)
        mask = mask.repeat(1, depth_values.size(1), 1, 1)
        log_dist_masked = torch.zeros_like(depth_values)
        log_dist = - conf * torch.abs(depth - depth_values)
        log_dist_masked[mask] = log_dist[mask]
        return log_dist_masked

    def forward(self, imgs, proj_matrices, depth_values, prior=None, depth_scale=1.0):

        # prior_depths, prior_confs, is_begin = prior if prior is not None else

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)
        dmap_min, dmap_max = depth_values[:, 0], depth_values[:, -1]

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        batch_size, height, width = imgs.size(0), imgs.size(3), imgs.size(4)

        outputs = {}
        depth, cur_depth = None, None

        for stage_idx in range(self.num_stage):
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
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=imgs[0].dtype,
                                                        device=imgs[0].device,
                                                        shape=[batch_size, height, width])

            # added by Khang
            depth_values_stage = F.interpolate(depth_range_samples.unsqueeze(1),
                                               [self.ndepths[stage_idx], height//int(stage_scale), width//int(stage_scale)],
                                               mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)

            log_likelihood = self.dnet(features_stage, proj_matrices_stage, depth_values=depth_values_stage,
                                       num_depth=self.ndepths[stage_idx],
                                       cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                       log=True, stage_idx=stage_idx, depth_min=dmap_min, depth_max=dmap_max)

            if prior is not None:
                prior_depths, prior_confs = prior[stage_name]
                est_prior_depth, est_prior_conf = self.pr_net(prior_depths, prior_confs)
                log_prior = self.get_log_prior(est_prior_depth, est_prior_conf, depth_values_stage / depth_scale)
            else:
                log_prior = 0.0
                est_prior_depth, est_prior_conf = None, None

            log_posterior = log_likelihood + log_prior
            posterior_vol = F.softmax(log_posterior, dim=1)
            depth = depth_regression(posterior_vol, depth_values_stage)
            final_conf = conf_regression(posterior_vol)
            var = torch.abs(depth.unsqueeze(1).detach() - depth_values_stage)
            var = depth_regression(posterior_vol, var)

            outputs_stage = {"depth": depth, "photometric_confidence": final_conf, "var": var,
                             "prior_depth": est_prior_depth, "prior_conf": est_prior_conf}
            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)
            # depth_range_values["stage{}".format(stage_idx + 1)] = depth_values_stage.detach()

        # all_depth_samples = get_depth_range_samples(depth_values, depth_values.size(1), depth_interval,
        #                                             device=imgs[0].device, dtype=imgs[0].dtype,
        #                                             shape=[batch_size, height, width]) # [B,D,H,W]
        # all_depth_samples = depth_values_stage

        total_dist = 0.0
        final_depth = depth.detach().unsqueeze(1)
        for stage_idx in range(self.num_stage):
            stage_name = "stage{}".format(stage_idx + 1)
            depth_stage, var_stage = outputs[stage_name]["depth"].detach(), outputs[stage_name]["var"].detach()
            depth_stage = F.interpolate(depth_stage.unsqueeze(1), [height, width], mode='nearest')
            var_stage = F.interpolate(var_stage.unsqueeze(1), [height, width], mode='nearest')
            dist = torch.exp(- (depth_stage - final_depth).abs() / (var_stage + 1e-16))
            total_dist = total_dist + dist
        total_dist /= self.num_stage
        #final_conf, indices = torch.max(total_dist, dim=1, keepdim=True)
        #final_depth = torch.gather(all_depth_samples, dim=1, index=indices)
        final_conf = total_dist
        feat_img = features[0]["stage3"].detach()
        final_depth /= depth_scale
        # depth map refinement
        if self.refine:
            final_depth, final_conf = self.refine_network(final_depth, final_conf, feat_img)
        else:
            final_depth = final_depth.squeeze(1)
            final_conf = final_conf.squeeze(1)
        outputs["final_depth"] = final_depth * depth_scale
        outputs["final_conf"] = final_conf

        return outputs
