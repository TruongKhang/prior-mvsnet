import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from models.losses import StereoFocalLoss
from .utils.warping import homo_warping_3D, resample_vol, homo_warping_2D, world_from_xy_depth
import numpy as np
from models.module import FCBlock, ResnetBlockFC
from models import hyperlayers
from .cvae import Encoder


Align_Corners_Range = False


class LatentScene(nn.Module):
    def __init__(self, in_channels, out_channels, num_instances, latent_dim,
                 num_hidden_units_phi, phi_layers, freeze_networks=False):
        super(LatentScene, self).__init__()

        # Auto-decoder: each scene instance gets its own code vector z
        self.latent_codes = nn.Embedding(num_instances, latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=latent_dim,
                                             hidden_ch=num_hidden_units_phi,
                                             num_hidden_layers=phi_layers - 2,
                                             in_ch=in_channels, out_ch=out_channels, outer_activation='sigmoid')
        if freeze_networks:
            for param in self.hyper_phi.parameters():
                param.requires_grad = False

        self.z = None

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        # if self.fit_single_srn:
        #     self.latent_reg_loss = 0
        # else:
        latent_reg_loss = torch.mean(self.z ** 2)

        return latent_reg_loss

    def get_occ_loss(self, inputs, mask=None, ndsamples=1):
        depth = inputs["depth"]
        depth_interval = np.random.randint(5, 10)
        depth_samples = get_depth_range_samples(cur_depth=depth,
                                                ndepth=ndsamples, depth_inteval_pixel=depth_interval,
                                                dtype=depth.dtype, device=depth.device,
                                                shape=[depth.size(0), depth.size(1), depth.size(2)])
        inputs["depth"] = depth_samples

        output = self.forward(inputs, ndsamples=ndsamples)
        output = output.squeeze(-1).view(depth.size(0), ndsamples, depth.size(1), depth.size(2))
        target = torch.zeros_like(output)
        target[:, (ndsamples-1)//2, :, :] = 1.0
        return output, target

    def forward(self, inputs, z=None, ndsamples=1):
        # self.logs = list()  # log saves tensors that"ll receive summaries when model"s write_updates function is called

        # Parse model input.
        instance_idcs = inputs["scene_idx"].long()
        pose = inputs["pose"]
        intrinsics = inputs["intrinsics"]
        uv = inputs["uv"].float()
        img_feat = inputs["img_feature"]
        depth = inputs["depth"]

        # if self.fit_single_srn:
        #     phi = self.phi
        # else:
        #     if self.has_params:  # If each instance has a latent parameter vector, we"ll use that one.
        if z is not None:
            self.z = z
        else:
            self.z = self.latent_codes(instance_idcs)

        phi = self.hyper_phi(self.z)  # Forward pass through hypernetwork yields a (callable) SRN.

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.

        # self.logs.extend(log)

        # Sapmle phi a last time at the final ray-marched world coordinates.
        points_xyz = world_from_xy_depth(uv, depth, pose, intrinsics)
        points_xyz = points_xyz.contiguous()
        points_xyz = points_xyz.view(points_xyz.size(0), -1, points_xyz.size(-1))

        batch_size, nchannels, h, w = img_feat.size()
        img_feat = img_feat.permute(0, 2, 3, 1)
        img_feat = img_feat.view(batch_size, -1, nchannels).repeat(1, ndsamples, 1)

        occ = phi(torch.cat((points_xyz, img_feat), dim=-1))

        # Translate features at ray-marched world coordinates to RGB colors.
        # novel_views = self.pixel_generator(v)
        # if not self.fit_single_srn:
        #     self.logs.append(("embedding", "", self.latent_codes.weight, 500))
        #     self.logs.append(("scalar", "embed_min", self.z.min(), 1))
        #     self.logs.append(("scalar", "embed_max", self.z.max(), 1))

        return occ


class DepthNet(nn.Module):
    def __init__(self, ndepths):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                prev_state=None, stage_idx=None):
        # proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices[1])+1, "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices #proj_matrices[0], proj_matrices[1:]

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
        # if stage_idx > 0:
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
           prob_volume_pre += prob_volume_init

        # log_prob_volume = F.log_softmax(prob_volume_pre, dim=1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)

        # itg_prob_volume = prob_volume

        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 2 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=[0, 0, 0, 0, 0, 1]), (2, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        # else:
        #     itg_prob_volume = cost_reg
        #     depth, photometric_confidence = None, None
        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume}

        # return {"depth": depth,  "photometric_confidence": photometric_confidence}


class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8), is_traning=True):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.is_training = is_traning
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
            last_layer = [True, True, True]
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i], last_layer=last_layer[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet(self.ndepths)
        self.latent_scene = LatentScene(3+self.feature.out_channels[-1], 1, 119, 64, 64, 5)

        if self.is_training:
            all_params = list(self.feature.parameters()) + list(self.cost_regularization.parameters()) + \
                         list(self.latent_scene.hyper_phi.parameters())
            for param in all_params:
                param.requires_grad = False

    def forward(self, imgs, proj_matrices, depth_values, first_view=None,
                gt_depth=None, gt_mask=None, scene_ids=None):

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
        origin_img_feat = features[0]["stage{}".format(self.num_stage)].detach()

        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            # edit by Khang
            proj_matrices_stage = torch.unbind(proj_matrices_stage, 1)
            refs, srcs = proj_matrices_stage[0], proj_matrices_stage[1:]
            intrinsics, extrinsics = refs[:, 1, :3, :3], refs[:, 0, :4, :4]
            gt_depth_stage = gt_depth["stage{}".format(stage_idx + 1)] if gt_depth is not None else None
            gt_mask_stage = gt_mask["stage{}".format(stage_idx + 1)] if gt_mask is not None else None

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [img.shape[2], img.shape[3]],
                                          mode='bilinear', align_corners=Align_Corners_Range).squeeze(1)
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
                                               [self.ndepths[stage_idx], img.shape[2]//int(stage_scale),
                                                img.shape[3]//int(stage_scale)], mode='trilinear',
                                               align_corners=Align_Corners_Range).squeeze(1)

            height, width = img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)
            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=imgs.device),
                                   torch.arange(0, width, dtype=torch.float32, device=imgs.device)])
            y, x = y.contiguous(), x.contiguous()
            uv = torch.stack([x, y], dim=-1)
            uv = uv.unsqueeze(0).unsqueeze(0)

            img_feat = F.interpolate(origin_img_feat, [height, width], mode='bilinear', align_corners=Align_Corners_Range)

            prior_vol = torch.zeros_like(depth_values_stage) + 1e-6

            with torch.no_grad():
                if first_view.sum().item() < first_view.size(0):
                    pose = torch.inverse(extrinsics[~first_view])
                    img_feat_pred = img_feat[~first_view]
                    subset_depth_values = depth_values_stage[~first_view]
                    pred_vol = self.latent_scene({"scene_idx": scene_ids[~first_view], "pose": pose,
                                                  "depth": subset_depth_values, "intrinsics": intrinsics[~first_view],
                                                  "uv": uv.repeat(subset_depth_values.size(0), self.ndepths[stage_idx], 1, 1, 1),
                                                  "img_feature": img_feat_pred}, ndsamples=self.ndepths[stage_idx])
                    pred_vol = pred_vol.squeeze(-1).view(subset_depth_values.size(0), self.ndepths[stage_idx], height, width)
                    prior_vol[~first_view] = pred_vol

            outputs_stage = self.DepthNet(features_stage, (refs, srcs),
                                          depth_values=depth_values_stage,
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                          stage_idx=stage_idx)

            prior_vol = F.normalize(prior_vol, p=1, dim=1)
            itg_cost_vol = outputs_stage["prob_volume"] * prior_vol
            itg_cost_vol = F.normalize(itg_cost_vol, p=1, dim=1)
            outputs_stage['depth'] = depth_regression(itg_cost_vol, depth_values=depth_values_stage)
            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 2 * F.avg_pool3d(F.pad(itg_cost_vol.unsqueeze(1), pad=[0, 0, 0, 0, 0, 1]),
                                                    (2, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(itg_cost_vol.detach(),
                                               depth_values=torch.arange(self.ndepths[stage_idx],
                                                                         device=itg_cost_vol.device, dtype=torch.float)).long()
                depth_index = depth_index.clamp(min=0, max=self.ndepths[stage_idx]-1)
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
            outputs_stage['photometric_confidence'] = photometric_confidence
            outputs_stage['prob_volume'] = itg_cost_vol

            depth = outputs_stage['depth']
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
            depth_range_values["stage{}".format(stage_idx + 1)] = depth_values_stage.detach()

            if stage_idx == 0:
                ndsamples = 9
                if self.is_training:
                    inp_depth = gt_depth_stage
                    inp_mask = (gt_mask_stage > 0.5)
                else:
                    inp_depth = depth
                    inp_mask = outputs_stage["photometric_confidence"] > 0.8
                img_feat_train = img_feat
                inp_latent_net = {"scene_idx": scene_ids, "pose": torch.inverse(extrinsics),
                                  "depth": inp_depth, "intrinsics": intrinsics,
                                  "uv": uv.repeat(img_feat.size(0), ndsamples, 1, 1, 1),
                                  "img_feature": img_feat_train}
                latent_out, latent_target = self.latent_scene.get_occ_loss(inp_latent_net, ndsamples=ndsamples)
                outputs["occ_output"] = latent_out
                outputs["occ_target"] = latent_target
                outputs["occ_mask"] = inp_mask.unsqueeze(1).repeat(1, ndsamples, 1, 1)
        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth
        outputs["depth_candidates"] = depth_range_values
        outputs['scene_embedding'] = self.latent_scene.z
        # outputs["total_loss_vol"] = total_loss
        return outputs
