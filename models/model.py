import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import get_depth_range_samples, depth_regression, FeatureNet, CostRegNet, conf_regression
from .utils.warping import homo_warping_3D, world_from_xy_depth
import numpy as np
from models import hyperlayers
from utils import tensor2numpy, save_images

np.random.seed(9121995)
Align_Corners_Range = False


class ImgEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimensions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(3, 32, 3, stride=2))
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(32, 64, 3, stride=2))
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 96, 3, stride=2))
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(96, 128, 3, stride=2))
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3, stride=2))
        self.fc_out = nn.Linear(128, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 128, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class LatentScene(nn.Module):
    def __init__(self, in_channels, out_channels, num_instances, latent_dim,
                 num_hidden_units_phi, phi_layers, is_training=True):
        super(LatentScene, self).__init__()

        # Auto-decoder: each scene instance gets its own code vector z
        self.latent_codes = nn.Embedding(num_instances, latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.1)

        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=latent_dim,
                                             hidden_ch=num_hidden_units_phi,
                                             num_hidden_layers=phi_layers - 2,
                                             in_ch=num_hidden_units_phi, out_ch=out_channels, outer_activation='linear')
        # self.img_encoder = ImgEncoder(c_dim=num_hidden_units_phi)
        # self.points_encoder = FCBlock(num_hidden_units_phi, 2, in_channels, num_hidden_units_phi)
        self.points_encoder = hyperlayers.HyperFC(hyper_in_ch=latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=latent_dim,
                                             hidden_ch=num_hidden_units_phi,
                                             num_hidden_layers=2,
                                             in_ch=in_channels, out_ch=num_hidden_units_phi, outer_activation='relu')
        if not is_training:
            for param in self.hyper_phi.parameters():
                param.requires_grad = False
            # for param in self.img_encoder.parameters():
            #     param.requires_grad = False
            for param in self.points_encoder.parameters():
                param.requires_grad = False

        # self.z = None

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        latent_reg_loss = torch.mean(self.z ** 2)

        return latent_reg_loss

    def get_occ_loss(self, inputs, mask=None, ndsamples=1, depth_min=0.0,
                     depth_max=1500.0, trans_vec=None, sampling_range=(20, 100, 5)):
        depth = inputs["depth"].unsqueeze(1)
        depth_samples = depth
        nr = (ndsamples - 1) // 2
        nl = ndsamples - 1 - nr
        min_r, max_r, inter_r = sampling_range
        if ndsamples > 1:
            # for i in range(nl):
            d_intervals = torch.tensor(np.random.choice(np.arange(min_r, max_r, inter_r), size=nl), device=depth.device).float()
            depth_samples = torch.cat((depth_samples, depth - d_intervals.view((1, nl, 1, 1))), dim=1)
            # for i in range(nr):
            d_intervals = torch.tensor(np.random.choice(np.arange(min_r, max_r, inter_r), size=nr), device=depth.device).float()
            depth_samples = torch.cat((depth_samples, depth + d_intervals.view((1, nr, 1, 1))), dim=1)
        inputs["depth"] = depth_samples.clamp(min=depth_min, max=depth_max)  # depth_samples
        inputs["depth"] = inputs["depth"][:, 1:, :, :]

        if mask is not None:
            inputs["depth"] = inputs["depth"] * mask.unsqueeze(1).float()
        target = torch.ones_like(inputs["depth"], requires_grad=False)
        target[:, :nl, :, :] = 0.0

        output, embeddings = self.forward(inputs, ndsamples=ndsamples - 1, trans_vec=trans_vec)
        output = output.squeeze(-1).view(inputs["depth"].size())

        return output, target, embeddings

    def predict_from_points(self, points_xyz, scene_ids, imgs):
        z = self.latent_codes(scene_ids)
        phi = self.hyper_phi(z)

        points_extractor = self.points_encoder(z)
        points_xyz = points_extractor(points_xyz)

        # img_feat = self.img_encoder(imgs)
        # img_feat = img_feat.unsqueeze(1)
        #
        # points_xyz = points_xyz + img_feat

        occ = phi(points_xyz)
        return occ

    def forward(self, inputs, ndsamples=1, trans_vec=None):

        # Parse model input.
        instance_idcs = inputs["scene_idx"].long()
        pose = inputs["pose"]
        intrinsics = inputs["intrinsics"]
        uv = inputs["uv"].float()
        img_feat = inputs["img_feature"]
        depth = inputs["depth"]

        z = self.latent_codes(instance_idcs)

        phi = self.hyper_phi(z)  # Forward pass through hypernetwork yields a (callable) SRN.

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.

        # Sapmle phi a last time at the final ray-marched world coordinates.
        points_xyz = world_from_xy_depth(uv, depth, pose, intrinsics)
        if trans_vec is None:
            trans_vec = torch.zeros_like(points_xyz)
        else:
            trans_vec = trans_vec.unsqueeze(1)
        points_xyz = points_xyz - trans_vec
        points_extractor = self.points_encoder(z)
        points_xyz = points_extractor(points_xyz)
        # points_xyz = self.points_encoder(points_xyz)

        # img_feat = self.img_encoder(img_feat)
        # img_feat = img_feat.unsqueeze(1)
        #
        # points_xyz = points_xyz + img_feat

        occ = phi(points_xyz)
        return occ, z


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                conv3d=True, log=False):
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

        # if not conv3d:
        #     volume_variance = volume_variance.squeeze(2)
        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        # if stage_idx > 0:
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
           prob_volume_pre += prob_volume_init

        # if conv3d:
        if log:
            prob_volume = F.log_softmax(prob_volume_pre, dim=1)
        else:
            prob_volume = F.softmax(prob_volume_pre, dim=1)
        # else:
        #     prob_volume = prob_volume_pre #torch.log(prob_volume_pre + 1e-6)
        # prob_volume = F.softmax(prob_volume_pre, dim=1)

        return prob_volume


class SeqProbMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8), is_training=True):
        super(SeqProbMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.is_training = is_training
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1": {
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

        self.cost_reg = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
                                       for i in range(self.num_stage)])
        self.DepthNet = DepthNet()
        self.latent_scene = LatentScene(3, 1, 128, 128, 256, 3, is_training=False)

        self.cost_reg_params = list(self.feature.parameters()) + list(self.cost_reg.parameters())

        if not self.is_training:
            for param in self.cost_reg_params:
                param.requires_grad = False
        # else:
        #     ckpt = torch.load('final_model_000049.ckpt')
        #     self.latent_scene.load_state_dict(ckpt['model'])
        #     for p in self.latent_scene.parameters():
        #         p.requires_grad = False

    def get_posterior(self, features, proj_matrices, depth_values, first_view=None, img_feat=None,
                      scene_ids=None, uv=None, scale=1.0, trans_vec=None, cost_reg=None):
        # depth_values = depth.unsqueeze(1)
        likelihood_vol = self.DepthNet(features, proj_matrices, depth_values, depth_values.size(1),
                                       cost_regularization=cost_reg, conv3d=False, log=True)
        refs, srcs = proj_matrices
        intrinsics, extrinsics = refs[:, 1, :3, :3].clone(), refs[:, 0, :4, :4]
        intrinsics[:, :2, :] /= scale

        prior_vol = torch.zeros_like(likelihood_vol)
        embeddings = []

        batch_size, ndepths, height, width = depth_values.size()
        if first_view.sum() < batch_size:
            with torch.no_grad():
                pose = torch.inverse(extrinsics)
                sub_bsize = batch_size - first_view.sum().item()
                depth_scaled = F.interpolate(depth_values, [height//int(scale), width//int(scale)], mode='nearest')
                pred_vol, embedding = self.latent_scene({"scene_idx": scene_ids[~first_view], "pose": pose[~first_view],
                                                  "depth": depth_scaled[~first_view], "intrinsics": intrinsics[~first_view],
                                                  "uv": uv.repeat(sub_bsize, ndepths, 1),
                                                  "img_feature": img_feat[~first_view]}, ndsamples=ndepths, trans_vec=trans_vec[~first_view])
                pred_vol = pred_vol.squeeze(-1).view(sub_bsize, ndepths, depth_scaled.size(2), depth_scaled.size(3))
                pred_vol = F.interpolate(pred_vol, [height, width], mode='nearest')
                prior_vol[~first_view] = -pred_vol**2 / 2
                embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0) if len(embeddings) > 0 else None
        prior_vol = prior_vol.clamp(min=-12, max=0)
        prior_conf, indices = torch.max(prior_vol, dim=1, keepdim=True)
        prior_depth = torch.gather(depth_values, 1, indices)
        # prior_vol = F.log_softmax(prior_vol, dim=1)
        # print(prior_vol.mean().item(), prior_vol.min().item(), prior_vol.max().item())
        itg_cost_vol = likelihood_vol + prior_vol
        # print(itg_cost_vol.min().item(), itg_cost_vol.min().item())
        return itg_cost_vol, embeddings, prior_depth.squeeze(1)

    def get_depth_candidates(self, cur_depth, stage_idx, depth_interval, shape, depth_min, depth_max):
        if cur_depth.dim() == 4:
            cur_depth = cur_depth.squeeze(1)
        batch_size = cur_depth.size(0)
        height, width = shape
        stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
        depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                      ndepth=self.ndepths[stage_idx],
                                                      depth_inteval_pixel=self.depth_interals_ratio[
                                                                              stage_idx] * depth_interval,
                                                      dtype=cur_depth.dtype, device=cur_depth.device,
                                                      shape=[batch_size, height, width],
                                                      max_depth=depth_max, min_depth=depth_min)
        # added by Khang
        depth_values_stage = F.interpolate(depth_range_samples.unsqueeze(1),
                                           [self.ndepths[stage_idx], height // int(stage_scale),
                                            width // int(stage_scale)], mode='trilinear',
                                           align_corners=Align_Corners_Range).squeeze(1)
        return depth_values_stage

    def forward(self, imgs, proj_matrices, depth_values, first_view=None, scene_ids=None,
                depth=None, iter=0.0, trans_vec=None):

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {"posterior": {}, "gt_posterior": {}, "scene_embedding": None}
        cur_depth = None
        height, width = imgs.shape[3], imgs.shape[4]
        img_feat = F.interpolate(imgs[:, 0], [height // 4, width // 4], mode='bilinear', align_corners=Align_Corners_Range)
        y, x = torch.meshgrid([torch.arange(0, height // 4, dtype=torch.float32, device=imgs.device),
                               torch.arange(0, width // 4, dtype=torch.float32, device=imgs.device)])
        y, x = y.contiguous().view(-1), x.contiguous().view(-1)
        uv = torch.stack([x, y], dim=-1)
        uv = uv.unsqueeze(0)

        if depth is None:
            stages = [0, 1]
        elif iter < 2:
            stages = [1]
        else:
            stages = [2]
        decay = iter if iter < 2 else iter - 2
        if decay > 2:
            decay = 2
        for stage_idx in stages:
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            # edit by Khang
            proj_matrices_stage = torch.unbind(proj_matrices_stage, 1)
            refs, srcs = proj_matrices_stage[0], proj_matrices_stage[1:]
            intrinsics, extrinsics = refs[:, 1, :3, :3].clone(), refs[:, 0, :4, :4]

            # t1 = time()
            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height, width],
                                          mode='bilinear', align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            new_depth_interval = depth_interval * np.power(2.0, -decay)
            depth_values_stage = self.get_depth_candidates(cur_depth, stage_idx, new_depth_interval, (height, width),
                                                           depth_min=depth_min, depth_max=depth_max)

            if stage_idx == 0:
                # t3 = time()
                likelihood_vol = self.DepthNet(features_stage, (refs, srcs),
                                               depth_values=depth_values_stage, num_depth=self.ndepths[stage_idx],
                                               cost_regularization=self.cost_reg[stage_idx], log=False)
                itg_cost_vol = likelihood_vol

                depth = depth_regression(itg_cost_vol, depth_values=depth_values_stage)
                conf = conf_regression(itg_cost_vol)

                outputs_stage = {"depth": depth, "photometric_confidence": conf, "prior_prob": None}
                outputs["stage{}".format(stage_idx + 1)] = outputs_stage
                outputs.update(outputs_stage)
            else:

                log_posterior, embeddings, prior_depth = self.get_posterior(features_stage, (refs, srcs), depth_values_stage,
                                                               first_view=first_view, img_feat=img_feat,
                                                               scene_ids=scene_ids, uv=uv, scale=4/stage_scale,
                                                               trans_vec=trans_vec, cost_reg=self.cost_reg[stage_idx])
                posterior = F.softmax(log_posterior, dim=1)

                # if self.is_training:
                #     conf, indices = torch.max(posterior, dim=1, keepdim=True)
                #     depth = depth_regression(posterior, depth_values=depth_values_stage)
                #     outputs_stage = {"depth": depth, "photometric_confidence": conf.squeeze(1), "prior_prob": None}
                # else:

                depth = depth_regression(posterior, depth_values=depth_values_stage)
                conf = conf_regression(posterior)

                depth_scaled = F.interpolate(depth.unsqueeze(1), [height // 4, width // 4], mode='nearest')
                conf_scaled = F.interpolate(conf.unsqueeze(1), [height // 4, width // 4], mode='nearest')

                depth_samples = depth_scaled
                d_intervals = torch.tensor(np.random.choice(np.arange(20, 100, 5), size=5),
                                           device=depth_scaled.device).float()
                depth_samples = torch.cat((depth_samples, depth_scaled - d_intervals.view((1, 5, 1, 1))), dim=1)
                depth_samples = depth_samples.clamp(min=depth_min, max=depth_max)  # depth_samples

                intrinsics[:, :2, :] = intrinsics[:, :2, :] * stage_scale / 4
                pose = torch.inverse(extrinsics)
                pred_vol, embeddings = self.latent_scene(
                    {"scene_idx": scene_ids, "pose": pose, "depth": depth_samples, "intrinsics": intrinsics,
                     "uv": uv.repeat(depth_samples.size(0), depth_samples.size(1), 1), "img_feature": img_feat},
                    ndsamples=depth_samples.size(1), trans_vec=trans_vec)
                pred_vol = pred_vol.squeeze(-1).view(depth_samples.size())

                outputs_stage = {"depth": depth, "photometric_confidence": conf,
                                 "scaled_conf": conf_scaled, "prior_prob": pred_vol}

                outputs["stage{}".format(stage_idx + 1)] = outputs_stage
                outputs.update(outputs_stage)
                outputs["scene_embedding"] = embeddings
                outputs["prior_depth"] = prior_depth

        return outputs, depth.detach()
