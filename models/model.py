import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from models.losses import StereoFocalLoss
from .utils.warping import homo_warping_3D, resample_vol, homo_warping_2D
from .utils.disp2prob import LaplaceDisp2Prob, OneHotDisp2Prob


Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self, ndepths):
        super(DepthNet, self).__init__()
        """vol_filtering_stage1 = nn.Sequential(nn.Conv2d(2*ndepths[0], ndepths[0], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[0]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[0], ndepths[0], kernel_size=3, bias=False, padding=1),
                                    nn.BatchNorm2d(ndepths[0]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[0], ndepths[0], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[0]),
                                    nn.Sigmoid())
        vol_filtering_stage2 = nn.Sequential(nn.Conv2d(2*ndepths[1], ndepths[1], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[1]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[1], ndepths[1], kernel_size=3, bias=False, padding=1),
                                    nn.BatchNorm2d(ndepths[1]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[1], ndepths[1], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[1]),
                                    nn.Sigmoid())
        vol_filtering_stage3 = nn.Sequential(nn.Conv2d(2*ndepths[2], ndepths[2], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[2]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[2], ndepths[2], kernel_size=3, bias=False, padding=1),
                                    nn.BatchNorm2d(ndepths[2]),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(ndepths[2], ndepths[2], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(ndepths[2]),
                                    nn.Sigmoid())
        self.vol_filtering = nn.ModuleList([vol_filtering_stage1, vol_filtering_stage2, vol_filtering_stage3])"""

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                prev_state=None, stage_idx=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        if stage_idx > 0:
            prob_volume_pre = cost_reg.squeeze(1)

        #if prob_volume_init is not None:
        #    prob_volume_pre += prob_volume_init

        # log_prob_volume = F.log_softmax(prob_volume_pre, dim=1)
            prob_volume = F.softmax(prob_volume_pre, dim=1)

        # edit by Khang
        # ref_proj_cur = ref_proj[:, 0].clone()
        # ref_proj_cur[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        # ref_proj_prev, prev_costvol, prev_depth_values, is_begin = prev_state
        # prev_depth, prev_cfd = prev_costvol
        # warped_depth, warped_cfd = homo_warping_2D(prev_depth.unsqueeze(1), prev_cfd.unsqueeze(1), ref_proj_prev, ref_proj_cur)
        # std = (1 - warped_cfd) + 1.2
        # mask = (warped_depth > 0.5).float()
        # warped_costvol = LaplaceDisp2Prob(depth_values, warped_depth, variance=std).getProb()
        # cur_vol = prob_volume.detach().clone()
        # if is_begin.sum() > 0:
        #     # print('is begining of video')
        #     B, D, H, W = prob_volume.size() #log_prob_volume.size()
        #     # prev_costvol = torch.zeros((B, D, H, W), dtype=torch.float32).cuda() #torch.log(torch.ones((B, D, H, W), dtype=torch.float32) / D).cuda()
        #     # if prev_costvol is None:
        #     #     prev_costvol = torch.zeros((B, D, H, W), dtype=torch.float32).cuda()
        #     # warped_costvol = resample_vol(prev_costvol, ref_proj_prev, ref_proj_cur, depth_values,
        #     #                                 prev_depth_values=prev_depth_values, begin_video=is_begin)
        #     warped_costvol[is_begin] = 0 #1.0 / D
        #     if is_begin.sum() < B:
        #         input_vol = torch.cat((cur_vol[~is_begin], warped_costvol[~is_begin]), dim=1)
        #         weight_vol = self.vol_filtering[stage_idx](input_vol)
        #         warped_costvol[~is_begin] = warped_costvol[~is_begin] * weight_vol
        # else:
        #     weight_vol = self.vol_filtering[stage_idx](torch.cat((cur_vol, warped_costvol), dim=1))
        #     warped_costvol = warped_costvol * weight_vol
        #
        # itg_prob_volume = prob_volume + warped_costvol * mask
        # itg_prob_volume = F.normalize(itg_prob_volume, p=1, dim=1)

            itg_prob_volume = prob_volume

            depth = depth_regression(itg_prob_volume, depth_values=depth_values)

            with torch.no_grad():
            # photometric confidence
                prob_volume_sum4 = 2 * F.avg_pool3d(F.pad(itg_prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 0, 1)), (2, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(itg_prob_volume, depth_values=torch.arange(num_depth, device=itg_prob_volume.device, dtype=torch.float)).long()
                depth_index = depth_index.clamp(min=0, max=num_depth-1)
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        else:
            itg_prob_volume = cost_reg
            depth, photometric_confidence = None, None
        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": itg_prob_volume}

        # return {"depth": depth,  "photometric_confidence": photometric_confidence}


class SeqProbMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8)):
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
            last_layer = [False, True, True]
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i], last_layer=last_layer[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet(self.ndepths)
        in_channels = (2, self.feature.out_channels[0])
        self.cvae = GenerationNet(input_channels=in_channels, output_channels=self.ndepths[0],
                                  num_filters=[16, 32, 64, 96, 128], latent_dim=8)
        self.f_comb = nn.Sequential(Conv3d(self.cr_base_chs[0]+1, self.cr_base_chs[0], kernel_size=1),
                                    Conv3d(self.cr_base_chs[0], self.cr_base_chs[0], kernel_size=3, padding=1),
                                    Conv3d(self.cr_base_chs[0], self.cr_base_chs[0], kernel_size=3, padding=1))
        #self.f_comb = Conv3d(self.cr_base_chs[0]*2, self.cr_base_chs[0], kernel_size=1)
        self.regress_prob = nn.Conv3d(self.cr_base_chs[0], 1, 1, bias=False)

    def forward(self, imgs, proj_matrices, depth_values, prev_state=None, gt_depth=None, gt_mask=None):

        prev_ref_matrices, prev_costvol, prev_depth_values, is_begin = prev_state

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}
        depth, cur_depth = None, None
        depth_range_values = {}
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            # edit by Khang
            prev_costvol_stage = prev_costvol["stage{}".format(stage_idx + 1)]
            prev_ref_matrix = prev_ref_matrices["stage{}".format(stage_idx + 1)]
            prev_depth_values_stage = prev_depth_values["stage{}".format(stage_idx + 1)]
            gt_depth_stage = gt_depth["stage{}".format(stage_idx + 1)].unsqueeze(1) if gt_depth is not None else None
            gt_mask_stage = gt_mask["stage{}".format(stage_idx + 1)].unsqueeze(1) if gt_mask is not None else None

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img.shape[2], img.shape[3]], mode='bilinear',
                                                align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            # added by Khang
            depth_values_stage = F.interpolate(depth_range_samples.unsqueeze(1),
                                                [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)],
                                                mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=depth_values_stage,
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                          prev_state=(prev_ref_matrix, prev_costvol_stage, prev_depth_values_stage, is_begin),
                                          stage_idx=stage_idx)

            if stage_idx == 0: #(self.num_stage-1):
                proj_matrices_stage = torch.unbind(proj_matrices_stage, 1)
                ref_proj = proj_matrices_stage[0]
                cur_ref_proj = ref_proj[:, 0].clone()
                cur_ref_proj[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                prev_depth, prev_cfd = prev_costvol_stage
                warped_depth, warped_cfd = homo_warping_2D(prev_depth.unsqueeze(1), prev_cfd.unsqueeze(1),
                                                           prev_ref_matrix, cur_ref_proj)
                if is_begin.sum() > 0:
                    costvol = F.softmax(
                        self.regress_prob(outputs_stage["prob_volume"].detach()[is_begin]).detach().squeeze(1), dim=1)
                    warped_depth[is_begin] = depth_regression(costvol, depth_values=depth_values_stage).unsqueeze(1)
                    prob_sum2 = 2 * F.avg_pool3d(F.pad(costvol.unsqueeze(1), pad=(0, 0, 0, 0, 0, 1)), (2, 1, 1),
                                                 stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(costvol, depth_values=torch.arange(self.ndepths[stage_idx],
                                                                                      device=costvol.device,
                                                                                      dtype=torch.float)).long()
                    depth_index = depth_index.clamp(min=0, max=self.ndepths[stage_idx] - 1)
                    warped_cfd[is_begin] = torch.gather(prob_sum2, 1, depth_index.unsqueeze(1))
                # gt_costvol = OneHotDisp2Prob(depth_values_stage, gt_depth_stage, variance=self.depth_interals_ratio[stage_idx]*depth_interval).getProb()
                # gt_costvol = gt_costvol + (1 - gt_mask) / self.ndepths[stage_idx]
                # sum_warped = warped_costvol.sum(dim=1)
                # sum_gt = gt_costvol.sum(dim=1)
                # print(sum_warped.min(), sum_warped.max())
                # print(sum_gt.min(), sum_gt.max())
                # h, w = imgs[:, 0].size(2), imgs[:, 0].size(3)
                # feature_img = F.interpolate(imgs[:, 0], [h//int(stage_scale), w//int(stage_scale)], mode='bilinear', align_corners=Align_Corners_Range)
                feature_img = features_stage[0].detach()

                if self.training:
                    gt_depth_stage = gt_depth_stage * gt_mask_stage
                    self.cvae(feature_img, warped_depth, warped_cfd, gt_depth_stage, gt_mask_stage)
                    recons_vol, kl_term = self.cvae.elbo(depth_values=depth_values_stage, variance=1)
                    # recons_vol = recons_vol.repeat(1, self.cr_base_chs[stage_idx], 1, 1, 1)
                    outputs_stage['kl'] = kl_term
                else:
                    self.cvae(feature_img, warped_depth, warped_cfd, training=False)
                    recons_vol = self.cvae.sample(testing=True, num_samples=self.cr_base_chs[stage_idx], depth_values=depth_values_stage, variance=1)
                itg_cost_vol = self.f_comb(torch.cat((outputs_stage["prob_volume"], recons_vol), dim=1)) #.squeeze(1) #outputs_stage["prob_volume"] + recons_vol
                itg_cost_vol = F.softmax(self.regress_prob(itg_cost_vol).squeeze(1), dim=1) #F.normalize(itg_cost_vol, p=1, dim=1)
                outputs_stage['depth'] = depth_regression(itg_cost_vol, depth_values=depth_values_stage)
                with torch.no_grad():
                    # photometric confidence
                    prob_volume_sum4 = 2 * F.avg_pool3d(F.pad(itg_cost_vol.unsqueeze(1), pad=(0, 0, 0, 0, 0, 1)),
                                                        (2, 1, 1), stride=1, padding=0).squeeze(1)
                    depth_index = depth_regression(itg_cost_vol.detach(),
                                                   depth_values=torch.arange(self.ndepths[stage_idx], device=itg_cost_vol.device,
                                                                             dtype=torch.float)).long()
                    depth_index = depth_index.clamp(min=0, max=self.ndepths[stage_idx]-1)
                    photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
                outputs_stage['photometric_confidence'] = photometric_confidence
                outputs_stage['prob_volume'] = itg_cost_vol
                # outputs_stage['kl'] = kl_term

            depth = outputs_stage['depth']
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
            depth_range_values["stage{}".format(stage_idx + 1)] = depth_values_stage.detach()
            # total_loss += outputs_stage["loss_vol"]

        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth
        outputs["depth_candidates"] = depth_range_values
        # outputs["total_loss_vol"] = total_loss

        return outputs
