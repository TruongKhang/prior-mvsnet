import torch
import torch.nn as nn
import torch.nn.functional as F


def seq_prob_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    depth_scale = kwargs.get("depth_scale", 1.0)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = 0.0

    for s in range(3): #(stage_inputs, stage_key) in [(inputs[k], k) for k in ["stage1", "stage2", "stage3"]]: # inputs.keys() if "stage" in k]:
        stage_key = "stage%d" % (s + 1)
        stage_inputs = inputs[stage_key]
        depth_est = stage_inputs["depth"] / depth_scale
        depth_gt = depth_gt_ms[stage_key] / depth_scale
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss
    last_stage = "stage4" if "stage4" in depth_gt_ms else "stage3"
    final_loss, depth_loss = masked_prob_loss(inputs["final_depth"] / depth_scale, inputs["final_conf"],
                                              (depth_gt_ms[last_stage] / depth_scale, mask_ms[last_stage] > 0.5))
    total_loss = total_loss + final_loss

    return total_loss, depth_loss


def masked_prob_loss(depth, var, target):
    gt_depths, valid_mask = target
    # valid_mask = valid_mask.float()
    # print(depth.shape, var.shape, gt_depths.shape, valid_mask.shape)
    gt = gt_depths[valid_mask]

    regl = torch.log(var + 1e-16)
    mean = depth[valid_mask]
    res = var[valid_mask]
    regl = regl[valid_mask]
    depth_error = torch.abs(gt - mean)
    final_loss = torch.mean(res * depth_error - regl) #torch.pow(gt - mean, 2) - regl)
    return final_loss, depth_error.mean()
