import torch
import torch.nn as nn
import torch.nn.functional as F


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    latent_loss = torch.mean(inputs["scene_embedding"] ** 2)
    occ_out, occ_target, occ_mask = inputs["occ_output"], inputs["occ_target"], inputs["occ_mask"]
    occ_mask = occ_mask.float()
    occ_loss = F.binary_cross_entropy(occ_out * occ_mask, occ_target * occ_mask)
    # print(occ_out.requires_grad, occ_target.requires_grad)
    total_loss = total_loss + latent_loss + occ_loss

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ["stage1", "stage2", "stage3"]]: # inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        """prob_volume_sum2 = 2 * F.avg_pool3d(F.pad(stage_inputs["prob_volume"].unsqueeze(1), pad=(0, 0, 0, 0, 0, 1)), (2, 1, 1), stride=1, padding=0).squeeze(1)
        depth_range_stage = inputs["depth_candidates"][stage_key]
        #depth_min = depth_range_stage[:, 0]
        #depth_interval = depth_range_stage[:, 1] - depth_min
        num_depth = depth_range_stage.size(1)
        depth_index = depth_regression(stage_inputs["prob_volume"].detach(), depth_values=torch.arange(num_depth, device=stage_inputs["prob_volume"].device, dtype=torch.float)).long()
        #depth_index = (depth_gt - depth_min) / depth_interval
        #depth_index = depth_index.long()
        depth_index = depth_index.clamp(min=0, max=num_depth-1)"""
        # gt_conf = stage_inputs["photometric_confidence"] #torch.gather(prob_volume_sum2, 1, depth_index.unsqueeze(1)).squeeze(1)
        # conf_loss = -torch.log(gt_conf[mask]).mean()
        # print(stage_key, depth_loss, conf_loss)

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss #+ conf_loss)
            if "kl" in stage_inputs:
                kl = torch.mean(stage_inputs["kl"])
                total_loss += depth_loss_weights[stage_idx] * kl
        else:
            total_loss += 1.0 * depth_loss #+ conf_loss)
            if "kl" in stage_inputs:
                kl = torch.mean(stage_inputs["kl"])
                total_loss += kl
        # if stage_key == 'stage3':
        #if "kl" in stage_inputs:
        #    kl = torch.mean(stage_inputs["kl"])
        #    stage_idx = 
        #    total_loss += depth_loss_weights[2] * kl
        # depth_values_stage = depth_values[stage_key]
        # est_prob_vol = stage_inputs["prob_volume"]
        # total_loss += stage_infos[stage_key]["loss_vol_weight"] * stereo_focal_loss.loss_per_level(est_prob_vol, depth_gt.unsqueeze(1), stage_infos[stage_key]["variance"], depth_values_stage)
    # total_loss += inputs["total_loss_vol"]
    # print(total_loss.item(), occ_loss.item(), occ_mask.sum())

    return total_loss, depth_loss


def scene_representation_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    latent_loss = torch.mean(inputs["scene_embedding"] ** 2)
    occ_out, occ_target, occ_mask = inputs["occ_output"], inputs["occ_target"], inputs["occ_mask"]
    # occ_mask = occ_mask.float()
    occ_loss = F.binary_cross_entropy_with_logits(occ_out[occ_mask], occ_target[occ_mask])
    # print(occ_out.requires_grad, occ_target.requires_grad)
    total_loss = total_loss + latent_loss + occ_loss
    return total_loss, occ_loss


# def seq_prob_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)
#
#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#
#     latent_loss = torch.mean(inputs["scene_embedding"] ** 2) if inputs["scene_embedding"] is not None else 0.0
#     total_loss = total_loss + latent_loss
#
#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         depth_est = stage_inputs["depth"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#
#         depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
#
#         if depth_loss_weights is not None:
#             stage_idx = int(stage_key.replace("stage", "")) - 1
#             total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
#         else:
#             total_loss += 1.0 * depth_loss
#     return total_loss, depth_loss

def seq_prob_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = 0.0 #torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    latent_loss = torch.mean(inputs["scene_embedding"] ** 2) if inputs["scene_embedding"] is not None else 0.0
    total_loss = total_loss + latent_loss
    if inputs["prior_prob"] is not None:
        conf = inputs["scaled_conf"]
        prior_loss = inputs["prior_prob"][:, 0, :, :] ** 2 / 2
        masked_prior_loss = prior_loss[conf > 0.8]
        if len(masked_prior_loss) > 0:
            total_loss = total_loss + torch.mean(masked_prior_loss)
            occ_out = inputs["prior_prob"][:, 1:, :, :]
            occ_target = torch.zeros_like(occ_out, requires_grad=False)
            occ_mask = (conf > 0.8).repeat(1, occ_out.size(1), 1, 1)
            occ_loss = F.binary_cross_entropy_with_logits(occ_out[occ_mask], occ_target[occ_mask])
            total_loss = total_loss + occ_loss

    depth_loss = 0.0
    mode = kwargs.get("mode")
    if mode != "test":
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            depth_est = stage_inputs["depth"]
            depth_gt = depth_gt_ms[stage_key]
            mask = mask_ms[stage_key]
            mask = mask > 0.5
            depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        #if stage_key == "stage3":
        #    if stage_inputs["label"] is not None:
        #        nll_loss_weight = kwargs.get("nll_loss_w", 1.0)
        #        prob_vol, indices = stage_inputs["prob_volume"], stage_inputs["label"]
        #        prob_loss = F.nll_loss(prob_vol*mask.unsqueeze(1).float(), indices) * nll_loss_weight
        #    else:
        #        prob_loss = 0
        #    total_loss = total_loss + prob_loss
        #else:
            if depth_loss_weights is not None:
                stage_idx = int(stage_key.replace("stage", "")) - 1
                total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss
    return total_loss, depth_loss
