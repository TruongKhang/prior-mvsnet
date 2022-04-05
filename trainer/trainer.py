import numpy as np
import os
import torch
import torch.nn.functional as F
import time

from base import BaseTrainer
from utils import AbsDepthError_metrics, Thres_metrics, tocuda, DictAverageMeter, inf_loop, tensor2float, tensor2numpy, save_images
from models.utils.warping import homo_warping_2D, get_prior


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None):
        super().__init__(model, criterion, optimizer, config, writer=writer)
        self.config = config
        self.data_loader = data_loader
        #self.data_loader.set_device(self.device)
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
        self.scale2dtu = config["trainer"]["scale2dtu"]
        self.num_stages = config["data_loader"]["args"]["num_stages"]
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

        # self.data_loader.dataset.generate_indices()
        # training
        for batch_idx, sample in enumerate(self.data_loader):
            start_time = time.time()

            # modified from the original by Khang
            sample_cuda = tocuda(sample)
            depth_gt_ms = sample_cuda["depth"]
            mask_ms = sample_cuda["mask"]
            num_stage = self.num_stages
            depth_gt = depth_gt_ms["stage{}".format(num_stage)]
            mask = mask_ms["stage{}".format(num_stage)]

            imgs, cam_params, depth_values = sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]
            # automatically compute depth scale
            d_interval = (depth_values[:, 1] - depth_values[:, 0]) / self.scale2dtu
            d_interval = d_interval.reshape(-1, 1, 1)

            depths, confs, masks = sample_cuda["prior_depths"], sample_cuda["prior_confs"], sample_cuda[
                    "prior_masks"]  # [B,N,1,H,W]
                # all_gt_depths = sample_cuda["gt_depths"]
                # for stage in cam_params.keys():
                #     cam_params_stage = cam_params[stage]
                #     # m = (masks[stage] > 0.5).float()
                #     warped_depths, warped_confs = homo_warping_2D(depths[stage], confs[stage], cam_params_stage)
                #     prior[stage] = warped_depths / self.depth_scale, warped_confs
            prior = get_prior(depths["stage{}".format(num_stage)], confs["stage{}".format(num_stage)], 
                              cam_params["stage{}".format(num_stage)], num_stages=num_stage) #, depth_scale=d_interval)

            self.optimizer.zero_grad()
            # for otm in self.optimizer:
            #     otm.zero_grad()

            outputs = self.model(imgs, cam_params, depth_values, prior=prior, depth_scale=d_interval.unsqueeze(1))

            loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=self.config["trainer"]["dlossw"],
                                              depth_scale=d_interval)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler["mvsnet"].step()

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
                        self.optimizer.param_groups[0]["lr"], loss, depth_loss, time.time() - start_time))
            # del scalar_outputs, image_outputs
            self.train_metrics.update({"loss": loss.item(), "depth_loss": depth_loss.item()}, n=depth_gt.size(0))

        # if "prior" in self.lr_scheduler.keys():
        #     self.lr_scheduler["prior"].step()

        if (epoch % self.config["trainer"]["eval_freq"] == 0) or (epoch == self.epochs - 1):
            self._valid_epoch(epoch)

        return self.train_metrics.mean()

    def _valid_epoch(self, epoch, save_folder='saved/'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
                                                                                     self.valid_data_loader.batch_size))
        # if save_folder is not None:
        #     path_depth = os.path.join(save_folder, 'depth_maps')
        #     if not os.path.exists(path_depth):
        #         os.makedirs(path_depth)
        #     path_cfd = os.path.join(save_folder, 'confidence')
        #     if not os.path.exists(path_cfd):
        #         os.makedirs(path_cfd)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                start_time = time.time()

                # modified from the original by Khang
                sample_cuda = tocuda(sample)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                num_stage = self.num_stages
                depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                mask = mask_ms["stage{}".format(num_stage)]

                imgs, cam_params, depth_values = sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]
                d_interval = (depth_values[:, 1] - depth_values[:, 0]) / self.scale2dtu
                d_interval = d_interval.reshape(-1, 1, 1)

                depths, confs = sample_cuda["prior_depths"], sample_cuda["prior_confs"]  # [B,N,1,H,W]
                    # all_gt_depths = sample_cuda["gt_depths"]
                    # masks = sample_cuda["prior_masks"]
                    # for stage in cam_params.keys():
                    #     cam_params_stage = cam_params[stage]
                    #     m = (masks[stage] > 0.5).float()
                    #     warped_depths, warped_confs = homo_warping_2D(depths[stage]*m, confs[stage]*m, cam_params_stage)
                    #     prior[stage] = warped_depths / self.depth_scale, warped_confs
                prior = get_prior(depths["stage{}".format(num_stage)], confs["stage{}".format(num_stage)], 
                                  cam_params["stage{}".format(num_stage)], num_stages=num_stage)

                outputs = self.model(imgs, cam_params, depth_values, prior=prior, depth_scale=d_interval.unsqueeze(1))

                loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms,
                                                  dlossw=self.config["trainer"]["dlossw"],
                                                  depth_scale=d_interval)

                depth_est = outputs["final_depth"].detach()
                #mvs_depth = outputs["depth"].detach()
                # vis_maps = vis_masks["stage3"] # outputs["vis_maps"]
                # folder = '%s/view%d' % (save_folder, batch_idx)
                # if not os.path.exists(folder):
                #     os.makedirs(folder)
                # ref_view = imgs[0, 0].permute(1, 2, 0).cpu().numpy()
                # plt.imsave('%s/ref_view.png' % save_folder, ref_view)
                # for idx in range(vis_maps.size(1)):
                #     vmap = vis_maps[0, idx].squeeze(0).cpu().numpy()
                #     vmap = (vmap * 255).astype(np.uint8)
                #     plt.imsave('%s/vis_map_src%d.png' % (folder, idx+1), vmap)
                #     src_view = imgs[0, idx+1].permute(1, 2, 0).cpu().numpy()
                #     plt.imsave('%s/src_view%d.png' % (save_folder, idx+1), src_view)
                di = d_interval[0].item()
                scalar_outputs = {"loss": loss,
                                  "depth_loss": depth_loss,
                                  "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                                  #"depth_error_no_refinement": AbsDepthError_metrics(mvs_depth, depth_gt, mask > 0.5),
                                  "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*2),
                                  "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*4),
                                  "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*8),
                                  "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*14),
                                  "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*20),

                                  "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di*2.0]),
                                  "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [di*2.0, di*4.0]),
                                  "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [di*4.0, di*8.0]),
                                  "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [di*8.0, di*14.0]),
                                  "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [di*14.0, di*20.0]),
                                  "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                               [di*20.0, 1e5]),
                                  }

                """prior_depth_est = outputs["prior_depth"].squeeze(1)
                image_outputs = {"depth_est": depth_est * mask,
                                 "depth_est_nomask": depth_est,
                                 "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
                                 "ref_img": sample_cuda["imgs"][:, 0].cpu(),
                                 "mask": sample_cuda["mask"]["stage1"].cpu(),
                                 "errormap": (depth_est - depth_gt).abs() * mask,
                                 "prior_depth": prior_depth_est * mask,
                                 "error_prior_depth": (prior_depth_est - depth_gt).abs() * mask}
                save_images(self.writer, 'val', tensor2numpy(image_outputs), batch_idx)"""

                if batch_idx % self.log_step == 0:
                    # save_scalars(logger, 'test', scalar_outputs, global_step)
                    # save_images(logger, 'test', image_outputs, global_step)
                    print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                        epoch, self.epochs, batch_idx, len(self.valid_data_loader), loss, scalar_outputs["depth_loss"],
                        time.time() - start_time))
                self.valid_metrics.update(tensor2float(scalar_outputs))
                del scalar_outputs  # , image_outputs

        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", self.valid_metrics.mean())

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
