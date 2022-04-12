import torch
import time

from base import BaseTrainer
from utils import AbsDepthError_metrics, Thres_metrics, tocuda, DictAverageMeter, inf_loop, tensor2float
from models.utils.warping import get_prior


class PriornetTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None):
        super().__init__(model, criterion, optimizer, config, writer=writer)
        self.config = config
        self.data_loader = data_loader
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
            depth_gt_ms, mask_ms = sample_cuda["depth"], sample_cuda["mask"]
            stage_key = "stage{}".format(self.num_stages)
            depth_gt, mask = depth_gt_ms[stage_key], mask_ms[stage_key]

            cam_params, depth_values = sample_cuda["proj_matrices"], sample_cuda["depth_values"]

            depths, confs = sample_cuda["prior_depths"], sample_cuda["prior_confs"]  # [B,N,1,H,W]]
            prior = get_prior(depths[stage_key], confs[stage_key], cam_params[stage_key], num_stages=self.num_stages, thres_view=1) #, depth_scale=d_interval)
            # ref_depth, ref_conf = depths[:, 0, ...], confs[:, 0, ...]
            w_src_depths, w_src_confs = prior[stage_key]
            self.optimizer.zero_grad()

            # scale depth to DTU range, automatically compute depth scale
            depth_scale = (depth_values[:, 1] - depth_values[:, 0]) / self.scale2dtu
            depth_scale, depth_min = depth_scale.reshape(-1, 1, 1), depth_values[:, 0].reshape(-1, 1, 1)
            w_src_depths = (w_src_depths - depth_min.unsqueeze(1).unsqueeze(1)) / depth_scale.unsqueeze(1).unsqueeze(1)
            nonzero_mask = w_src_depths > 0
            w_src_depths[~nonzero_mask] = 0
            w_src_depths[nonzero_mask] += 425

            pred_depth, pred_conf = self.model(w_src_depths, w_src_confs)

            depth_gt = (depth_gt - depth_min) / depth_scale + 425
            target = depth_gt.unsqueeze(1), (mask > 0.5).unsqueeze(1)
            loss, _ = self.criterion(pred_depth, pred_conf, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                    epoch, self.epochs, batch_idx, len(self.data_loader), self.optimizer.param_groups[0]["lr"], loss,
                    AbsDepthError_metrics(pred_depth.squeeze(1), depth_gt, mask > 0.5), time.time() - start_time))
            # del scalar_outputs, image_outputs
            self.train_metrics.update({"loss": loss.item(),
                                       "depth_loss": AbsDepthError_metrics(pred_depth.squeeze(1), depth_gt, mask > 0.5).item()},
                                      n=depth_gt.size(0))

        self.lr_scheduler.step()

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

        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                start_time = time.time()

                # modified from the original by Khang
                sample_cuda = tocuda(sample)
                depth_gt_ms, mask_ms = sample_cuda["depth"], sample_cuda["mask"]
                stage_key = "stage{}".format(self.num_stages)
                depth_gt, mask = depth_gt_ms[stage_key], mask_ms[stage_key]

                cam_params, depth_values = sample_cuda["proj_matrices"], sample_cuda["depth_values"]
                depths, confs = sample_cuda["prior_depths"], sample_cuda["prior_confs"]  # [B,N,1,H,W]
                prior = get_prior(depths[stage_key], confs[stage_key], cam_params[stage_key], num_stages=self.num_stages, thres_view=1)

                w_src_depths, w_src_confs = prior[stage_key]
                # scale depth to DTU range, automatically compute depth scale
                depth_scale = (depth_values[:, 1] - depth_values[:, 0]) / self.scale2dtu
                depth_scale, depth_min = depth_scale.reshape(-1, 1, 1), depth_values[:, 0].reshape(-1, 1, 1)
                w_src_depths = (w_src_depths - depth_min.unsqueeze(1).unsqueeze(1)) / depth_scale.unsqueeze(1).unsqueeze(1)
                nonzero_mask = w_src_depths > 0
                w_src_depths[~nonzero_mask] = 0
                w_src_depths[nonzero_mask] += 425

                pred_depth, pred_conf = self.model(w_src_depths, w_src_confs)

                depth_gt = (depth_gt - depth_min) / depth_scale + 425
                target = depth_gt.unsqueeze(1), (mask > 0.5).unsqueeze(1)
                loss, _ = self.criterion(pred_depth, pred_conf, target)

                depth_est = pred_depth.squeeze(1).detach()
                ref_depth = (depths[stage_key][:, 0, ...].squeeze(1) - depth_min) / depth_scale
                nonzero_mask = ref_depth > 0
                ref_depth[~nonzero_mask] = 0
                ref_depth[nonzero_mask] += 425

                di = depth_scale[0].item()
                scalar_outputs = {"loss": loss,
                                  "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                                  "depth_error_input": AbsDepthError_metrics(ref_depth, depth_gt, mask > 0.5),
                                  "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                                  "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                                  "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                                  "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                                  "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                                  # "thres2mm_error_input": Thres_metrics(ref_depth, depth_gt, mask > 0.5, di * 2),
                                  # "thres4mm_error_input": Thres_metrics(ref_depth, depth_gt, mask > 0.5, di * 4),
                                  # "thres8mm_error_input": Thres_metrics(ref_depth, depth_gt, mask > 0.5, di * 8),
                                  # "thres14mm_error_input": Thres_metrics(ref_depth, depth_gt, mask > 0.5, di * 14),
                                  # "thres20mm_error_input": Thres_metrics(ref_depth, depth_gt, mask > 0.5, di * 20),

                                  "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                                  "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [2.0, 4.0]),
                                  "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [4.0, 8.0]),
                                  "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [8.0, 14.0]),
                                  "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [14.0, 20.0]),
                                  "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                               [20.0, 1e5]),
                                  }

                if batch_idx % self.log_step == 0:
                    # save_scalars(logger, 'test', scalar_outputs, global_step)
                    # save_images(logger, 'test', image_outputs, global_step)
                    print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                        epoch, self.epochs, batch_idx, len(self.valid_data_loader), loss,
                        scalar_outputs["abs_depth_error"], time.time() - start_time))
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
