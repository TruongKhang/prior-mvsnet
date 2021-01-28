import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets.data_loaders as module_data
import models.model as module_arch
import models.losses as module_loss
from trainer import Trainer


SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True


def main(config):
    logger = config.get_logger('train')

    data_loader = config.init_obj('data_loader', module_data)
    # setup data_loader instances
    init_kwags = {
        "data_path": config["data_loader"]["args"]["data_path"],
        "data_list": "lists/dtu/val.txt",
        "mode": "val",
        "num_srcs": config["data_loader"]["args"]["num_srcs"],
        "num_depths": config["data_loader"]["args"]["num_srcs"],
        "interval_scale": config["data_loader"]["args"]["interval_scale"],
        "shuffle": False,
        "seq_size": config["data_loader"]["args"]["seq_size"],
        "batch_size": 1
    }
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(**init_kwags)

    # build models architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    """print('Load pretrained model')
    checkpoint = torch.load('pretrained_model_kitti2.pth')
    new_state_dict = {}
    for key, val in checkpoint['state_dict'].items():
        new_state_dict[key.replace('module.', '')] = val
    model.load_state_dict(new_state_dict, strict=False)
    print('Done')"""

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, optimizer,
                           config=config,
                           data_loader=data_loader,
                           valid_data_loader=valid_data_loader,
                           lr_scheduler=lr_scheduler)

    trainer.train()


# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        TrainImgLoader.dataset.generate_indices()
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            # modified from the original by Khang
            sample = tocuda(sample)
            is_begin = sample['is_begin'].type(torch.bool)

            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample,
                                                               is_begin, args, global_step=global_step)

            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    # save_scalars(logger, 'train', scalar_outputs, global_step)
                    # save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           scalar_outputs['depth_loss'],
                           time.time() - start_time))
                del scalar_outputs #, image_outputs

        # checkpoint
        chkpt_file = "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx)
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, chkpt_file)
        gc.collect()
        # chkpt_file = "{}/model_{:0>6}.ckpt".format(args.logdir, 15)

        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            val_model = SeqProbMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                                      depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                                      share_cr=args.share_cr,
                                      cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                                      grad_method=args.grad_method, is_training=False)
            val_model.to(torch.device(args.device))
            val_model_loss = seq_prob_loss

            val_optimizer = optim.Adam(filter(lambda p: p.requires_grad, val_model.latent_scene.parameters()),
                                       lr=0.0001, betas=(0.9, 0.999), weight_decay=args.wd)
            # load checkpoint file specified by args.loadckpt
            print("loading model {}".format(chkpt_file))
            val_state_dict = torch.load(chkpt_file)
            new_state_dict = {}
            for k, v in val_state_dict['model'].items():
                new_state_dict[k.replace('module.', '')] = v
            val_model.load_state_dict(new_state_dict)

            validate(val_model, val_model_loss, val_optimizer, TestImgLoader, epoch_idx, args)
            del val_model, val_optimizer


def validate(model, model_loss, optimizer, ValImgLoader, epoch_idx, args):
    # testing
    avg_test_scalars = DictAverageMeter()
    cnt = 0
    # name_params = [n for n, p in model.named_parameters() if p.requires_grad]
    # print("Trainable parameters: ", len(name_params))
    # print(name_params)
    for batch_idx, sample in enumerate(ValImgLoader):
        start_time = time.time()
        global_step = len(ValImgLoader) * epoch_idx + batch_idx
        do_summary = global_step % args.summary_freq == 0

        # modified from the original by Khang
        sample = tocuda(sample)
        is_begin = sample['is_begin'].type(torch.bool)
        # cnt = 0 if is_begin.item() == 1 else cnt + 1
        # if cnt >= args.few_shots:
        #     loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, is_begin, args)
        # else:
        #     loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample,
        #                                                        is_begin, args, is_testing=True)
        loss, scalar_outputs, image_outputs = val_sample(model, model_loss, optimizer, sample, is_begin,
                                                         args, global_step=global_step)

        if (not is_distributed) or (dist.get_rank() == 0):
            if do_summary:
                # save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
                print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                    epoch_idx, args.epochs,
                    batch_idx,
                    len(ValImgLoader), loss,
                    scalar_outputs["depth_loss"],
                    time.time() - start_time))
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs #, image_outputs

    if (not is_distributed) or (dist.get_rank() == 0):
        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
    gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample_cuda, is_begin, args, is_testing=False, global_step=0):
    model.train()

    depth_est, loss, depth_loss = None, None, None

    # sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)] * args.depth_scale
    mask = mask_ms["stage{}".format(num_stage)]

    for t in range(4):
        # print(t)
        optimizer.zero_grad()
        # prev_ref_matrices, itg_vol = prev_state

        outputs, depth_est = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"],
                                   first_view=is_begin, scene_ids=sample_cuda["scene_idx"], depth=depth_est,
                                   iter=t, trans_vec=sample_cuda["trans_norm"])

        loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms,
                                      dlossw=[float(e) for e in args.dlossw.split(",") if e], training=(not is_testing))
        loss.backward()

        optimizer.step()

    # depth_est = model.depth_estimation.depth.data * args.depth_scale
    # depth_loss = F.smooth_l1_loss(depth_est[mask > 0.5], depth_gt[mask > 0.5], reduction='mean')

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)}

    # image_outputs = {"depth_est": depth_est * mask,
    #                  "depth_est_nomask": depth_est,
    #                  "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
    #                  "ref_img": sample_cuda["imgs"][:, 0].cpu(),
    #                  "mask": sample_cuda["mask"]["stage1"].cpu(),
    #                  "errormap": (depth_est - depth_gt).abs() * mask,
    #                  }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), None #tensor2numpy(image_outputs)


def val_sample(model, model_loss, optimizer, sample_cuda, is_begin, args, global_step=None):
    if num_gpus > 1:
        model_val = model.module
    else:
        model_val = model

    depth_est, loss, depth_loss = None, None, None

    model_val.train()
    optimizer.zero_grad()
    #model_val.eval()

    # sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)] * args.depth_scale
    mask = mask_ms["stage{}".format(num_stage)]
    for t in range(5):
        outputs, depth_est = model_val(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"],
                                       first_view=is_begin, scene_ids=sample_cuda["scene_idx"], depth=depth_est,
                                       trans_vec=sample_cuda["trans_norm"], iter=t)

        loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms,
                                  dlossw=[float(e) for e in args.dlossw.split(",") if e], training=False)
        loss.backward()
        optimizer.step()

        save_images(logger, 'val', tensor2numpy({"depth_est_%d" % t: depth_est,
                                                 "prior_depth_%d" % t: outputs["prior_depth"]}), global_step)

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
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
                     "ref_img": sample_cuda["imgs"][:, 0].cpu(),
                     "mask": sample_cuda["mask"]["stage1"].cpu(),
                     "errormap": (depth_est - depth_gt).abs() * mask}

    # if is_distributed:
    #     scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample_cuda, is_begin, args):
    if num_gpus > 1:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    # sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)] * args.depth_scale
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"],
                         first_view=is_begin, gt_depth=depth_gt_ms, gt_mask=mask_ms, scene_ids=sample_cuda["scene_idx"])
    depth_est = outputs["depth"].detach() * args.depth_scale

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

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
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
                     "ref_img": sample_cuda["imgs"][:, 0].cpu(),
                     "mask": sample_cuda["mask"]["stage1"].cpu(),
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        # test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        print(args.loadckpt)
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = SeqProbMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method)
    model.to(device)
    ckpt = torch.load('pretrained_model.ckpt')
    model.load_state_dict(ckpt['model'])
    model_loss = seq_prob_loss #cas_mvsnet_loss

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict['model'].items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale,
                               shuffle=True, seq_size=7, batch_size=args.batch_size, depth_scale=args.depth_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "val", 3, args.numdepth, args.interval_scale,
                              shuffle=False, seq_size=7, batch_size=1, depth_scale=args.depth_scale)

    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    TrainImgLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                num_workers=4, pin_memory=args.pin_m)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    TestImgLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=2,
                               pin_memory=args.pin_m)

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
