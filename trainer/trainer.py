import numpy as np
import os
import torch
import time

from base import BaseTrainer
from utils import AbsDepthError_metrics, Thres_metrics, tocuda, DictAverageMeter, inf_loop
from .data_structure import PriorState
from models.utils.warping import homo_warping_2D


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None):
        super().__init__(model, criterion, optimizer, config, writer=writer)
        self.config = config
        self.data_loader = data_loader
        self.data_loader.set_device(self.device)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['logging_every'] # int(np.sqrt(data_loader.batch_size))
        self.depth_scale = config["trainer"]["depth_scale"]
        self.use_prior = config["trainer"]["use_prior"]
        self.train_metrics = DictAverageMeter()
        self.valid_metrics = DictAverageMeter()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        print('Epoch {}:'.format(epoch))

        self.data_loader.dataset.generate_indices()
        prior_state = PriorState(max_size=4)
        # training
        for batch_idx, sample in enumerate(self.data_loader):
            start_time = time.time()

            # modified from the original by Khang
            sample_cuda = tocuda(sample)
            is_begin = sample_cuda['is_begin'].type(torch.uint8)
            depth_gt_ms = sample_cuda["depth"]
            mask_ms = sample_cuda["mask"]
            num_stage = len(self.config["arch"]["args"]["ndepths"])
            depth_gt = depth_gt_ms["stage{}".format(num_stage)] * self.depth_scale
            mask = mask_ms["stage{}".format(num_stage)]

            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

            if is_begin.sum() < len(is_begin):
                prior_state.reset()
            prior = None
            if self.use_prior:
                prior = {}
                if self.config["dataset_name"] == 'dtu':
                    depths, confs = sample_cuda["prior_depths"], sample_cuda["prior_confs"] # [B,N,1,H,W]
                    for stage in cam_params.keys():
                        cam_params_stage = cam_params[stage]
                        warped_depths, warped_confs = homo_warping_2D(depths[stage], confs[stage], cam_params_stage)
                        prior[stage] = warped_depths / self.depth_scale, warped_confs
                else:
                    if prior_state.size() == 4:
                        depths, confs, proj_matrices = prior_state.get()
                        for stage in depths.keys():
                            warped_depths, warped_confs = homo_warping_2D(depths, confs, proj_matrices, ref_proj=cam_params)
                            prior[stage] = warped_depths / self.depth_scale, warped_confs
                    else:
                        prior = None

            # self.optimizer.zero_grad()
            for otm in self.optimizer:
                otm.zero_grad()

            outputs = self.model(imgs, cam_params, sample_cuda["depth_values"], prior=prior,
                                 depth_scale=self.depth_scale)

            loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=self.config["trainer"]["dlossw"])
            loss.backward()
            for otm in self.optimizer:
                otm.step()
            self.lr_scheduler["mvsnet"].step()
            if self.config["dataset_name"] != 'dtu':
                depth_est, conf_est = {}, {}
                for i in range(num_stage):
                    stage = "stage%d" % (i+1)
                    depth_est[stage] = outputs[stage]["depth"].detach()
                    conf_est[stage] = outputs[stage]["photometric_confidence"].detach()
                prior_state.update(depth_est, conf_est, cam_params)

            # scalar_outputs = {"loss": loss,
            #                   "depth_loss": depth_loss,
            #                   "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
            #                   "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
            #                   "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
            #                   "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)}

            # image_outputs = {"depth_est": depth_est * mask,
            #                  "depth_est_nomask": depth_est,
            #                  "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
            #                  "ref_img": sample_cuda["imgs"][:, 0].cpu(),
            #                  "mask": sample_cuda["mask"]["stage1"].cpu(),
            #                  "errormap": (depth_est - depth_gt).abs() * mask,
            #                  }

            if batch_idx % self.log_step == 0:
                # save_scalars(self.writer, 'train', scalar_outputs, global_step)
                # save_images(self.writer, 'train', image_outputs, global_step)
                print(
                    "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                        epoch, self.epochs, batch_idx, len(self.data_loader),
                        self.optimizer[0].param_groups[0]["lr"], loss, depth_loss, time.time() - start_time))
            # del scalar_outputs, image_outputs
            self.train_metrics.update({"loss": loss.item(), "depth_loss": depth_loss.item()}, n=depth_gt.size(0))

        if (epoch % self.config["trainer"]["eval_freq"] == 0) or (epoch == self.epochs - 1):
            self._valid_epoch(epoch)

        #if self.lr_scheduler is not None:
        #    for lrs in self.lr_scheduler:
        #        lrs.step()
        if "prior" in self.lr_scheduler.keys():
            self.lr_scheduler.step()

        return self.train_metrics.mean()

    def _valid_epoch(self, epoch, save_folder=None):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
                                                                                     self.valid_data_loader.batch_size))
        if save_folder is not None:
            path_depth = os.path.join(save_folder, 'depth_maps')
            if not os.path.exists(path_depth):
                os.makedirs(path_depth)
            path_cfd = os.path.join(save_folder, 'confidence')
            if not os.path.exists(path_cfd):
                os.makedirs(path_cfd)

        self.model.eval()
        prior_state = PriorState(max_size=4)
        for batch_idx, sample in enumerate(self.valid_data_loader):
            start_time = time.time()

            # modified from the original by Khang
            sample_cuda = tocuda(sample)
            is_begin = sample['is_begin'].type(torch.uint8)
            depth_gt_ms = sample_cuda["depth"]
            mask_ms = sample_cuda["mask"]
            num_stage = len(self.config["arch"]["args"]["ndepths"])
            depth_gt = depth_gt_ms["stage{}".format(num_stage)] * self.depth_scale
            mask = mask_ms["stage{}".format(num_stage)]

            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            if is_begin.sum() < len(is_begin):
                prior_state.reset()
            prior = None
            if self.use_prior:
                prior = {}
                if self.config["dataset_name"] == 'dtu':
                    depths, confs = sample_cuda["prior_depths"], sample_cuda["prior_confs"]  # [B,N,1,H,W]
                    for stage in cam_params.keys():
                        cam_params_stage = cam_params[stage]
                        warped_depths, warped_confs = homo_warping_2D(depths[stage], confs[stage], cam_params_stage)
                        prior[stage] = warped_depths / self.depth_scale, warped_confs
                else:
                    if prior_state.size() == 4:
                        depths, confs, proj_matrices = prior_state.get()
                        for stage in depths.keys():
                            warped_depths, warped_confs = homo_warping_2D(depths, confs, proj_matrices,
                                                                          ref_proj=cam_params)
                            prior[stage] = warped_depths / self.depth_scale, warped_confs
                    else:
                        prior = None

            outputs = self.model(imgs, cam_params, sample_cuda["depth_values"], prior=prior,
                                 depth_scale=self.depth_scale)

            loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=self.config["trainer"]["dlossw"])

            if self.config["dataset_name"] != 'dtu':
                depth_est, conf_est = {}, {}
                for i in range(num_stage):
                    stage = "stage%d" % (i + 1)
                    depth_est[stage] = outputs[stage]["depth"].detach()
                    conf_est[stage] = outputs[stage]["photometric_confidence"].detach()
                prior_state.update(depth_est, conf_est, cam_params)

            depth_est = outputs["depth"].detach()

            scalar_outputs = {"loss": loss,
                              "depth_loss": depth_loss,
                              "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                              "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                              "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                              "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                              "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                              "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                              "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                              "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                              "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                              "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                              "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                          [14.0, 20.0]),
                              "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                           [20.0, 1e5]),
                              }

            # image_outputs = {"depth_est": depth_est * mask,
            #                  "depth_est_nomask": depth_est,
            #                  "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
            #                  "ref_img": sample_cuda["imgs"][:, 0].cpu(),
            #                  "mask": sample_cuda["mask"]["stage1"].cpu(),
            #                  "errormap": (depth_est - depth_gt).abs() * mask}

            if batch_idx % self.log_step == 0:
                # save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
                print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                    epoch, self.epochs, batch_idx, len(self.valid_data_loader), loss, scalar_outputs["depth_loss"],
                    time.time() - start_time))
            self.valid_metrics.update(scalar_outputs)
            del scalar_outputs  # , image_outputs

        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        # print("avg_test_scalars:", self.valid_metrics.mean())

        return self.valid_metrics.mean()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
