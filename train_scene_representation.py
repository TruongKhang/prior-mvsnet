import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from models.module import get_depth_range_samples
from utils import *
from models.losses import scene_representation_loss
import torch.distributed as dist
import math

SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = True
cudnn.deterministic = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,20,30,40:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--few_shots', type=int, default=0)
parser.add_argument('--depth_scale', type=float, default=1.0)


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = False


# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        """TrainImgLoader.dataset.generate_indices()
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            # modified from the original by Khang
            sample = tocuda(sample)
            is_begin = sample['is_begin'].type(torch.uint8)

            loss, min_val, max_val = train_sample(model, model_loss, optimizer, sample, is_begin, args)

            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', {"loss": loss}, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, min value = {:.3f}, max value = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           min_val, max_val,
                           time.time() - start_time))
        # checkpoint
        chkpt_file = "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx)
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, chkpt_file)
        gc.collect()"""
        chkpt_file = "final_model_{:0>6}.ckpt".format(49)

        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            val_model = LatentScene(3, 1, 128, 128, 128, 7, is_training=False)
            val_model.to(torch.device(args.device))
            val_model_loss = scene_representation_loss

            val_optimizer = optim.Adam(filter(lambda p: p.requires_grad, val_model.parameters()), lr=0.0001,
                                       betas=(0.9, 0.999), weight_decay=args.wd)
            # load checkpoint file specified by args.loadckpt
            print("loading model {}".format(chkpt_file))
            val_state_dict = torch.load(chkpt_file)
            val_model.load_state_dict(val_state_dict['model'])

            validate(val_model, val_model_loss, val_optimizer, TestImgLoader, epoch_idx, args)
            new_chkpt_file = "final_model_pSRN_{:0>6}.ckpt".format(epoch_idx)
            torch.save({'epoch': epoch_idx, 'model': val_model.state_dict()}, new_chkpt_file)


def validate(model, model_loss, optimizer, ValImgLoader, epoch_idx, args):
    # testing
    avg_test_scalars = DictAverageMeter()
    cnt = 0
    avg_loss = 0
    casmvs_model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                                 depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                                 share_cr=args.share_cr,
                                 cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                                 grad_method=args.grad_method)
    casmvs_ckpt = torch.load('casmvsnet.ckpt')
    casmvs_model.load_state_dict(casmvs_ckpt['model'])
    casmvs_model.to(torch.device(args.device))
    casmvs_model.eval()
    for p in casmvs_model.parameters():
        p.requires_grad = False
    #if (epoch_idx+1) % 10 == 0:
    #    test_epochs = 5
    #else:
    #    test_epochs = 1
    test_epochs = 10
    for e in range(test_epochs):
        avg_loss = 0
        for batch_idx, sample in enumerate(ValImgLoader):
            start_time = time.time()
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            # modified from the original by Khang
            sample = tocuda(sample)
            is_begin = sample['is_begin'].type(torch.uint8)
            #cnt = 0 if is_begin.item() == 1 else cnt + 1
            #if cnt >= args.few_shots:
            #    loss, min_val, max_val = test_sample_depth(model, model_loss, sample, is_begin, args)
            #else:
            loss, min_val, max_val = train_sample(model, model_loss, optimizer, sample,
                                                               is_begin, args, is_testing=True, casmvsnet=casmvs_model)

            if (not is_distributed) and (e == (test_epochs-1)):
                if do_summary:
                    #save_scalars(logger, 'val', {"loss": loss}, global_step)
                    print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, min value = {:.3f}, max value = {:.3f}, time = {:3f}".format(
                        epoch_idx, args.epochs,
                        batch_idx,
                        len(ValImgLoader), loss,
                        min_val, max_val,
                        time.time() - start_time))
            avg_loss += loss

        print("Test epoch %d, avg_loss: " %e, avg_loss / len(ValImgLoader))
    gc.collect()


def train_sample(model, model_loss, optimizer, sample_cuda, is_begin, args, is_testing=False, casmvsnet=None):
    model.train()
    optimizer.zero_grad()

    # sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage1"]
    mask = mask_ms["stage1"]

    height, width = depth_gt.shape[1], depth_gt.shape[2]
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_gt.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth_gt.device)])
    y, x = y.contiguous().view(-1), x.contiguous().view(-1)
    uv = torch.stack([x, y], dim=-1)
    uv = uv.unsqueeze(0)

    ndsamples = 5 #if not is_testing else 1

    proj_matrices = sample_cuda["proj_matrices"]["stage1"]
    proj_matrices = torch.unbind(proj_matrices, 1)
    refs, srcs = proj_matrices[0], proj_matrices[1:]
    intrinsics, extrinsics = refs[:, 1, :3, :3], refs[:, 0, :4, :4]
    img_feat = F.interpolate(sample_cuda["imgs"][:, 0], [height, width], mode='bilinear', align_corners=False)

    inp_latent_net = {"scene_idx": sample_cuda["scene_idx"], "pose": torch.inverse(extrinsics),
                      "depth": depth_gt, "intrinsics": intrinsics,
                      "uv": uv,
                      "img_feature": img_feat}

    if not is_testing:
        inp_latent_net["uv"] = uv.repeat(img_feat.size(0), ndsamples-1, 1)
        latent_out, latent_target, embeddings = model.get_occ_loss(inp_latent_net, mask=mask,
                                                                   ndsamples=ndsamples, is_training=(not is_testing), trans_vec=sample_cuda["trans_norm"])

        inp_latent_net["uv"] = uv.repeat(img_feat.size(0), 1, 1)
        inp_latent_net["depth"] = depth_gt.unsqueeze(1)
        gt_prob, _ = model(inp_latent_net, ndsamples=1)
        gt_prob = gt_prob.squeeze(-1).view(depth_gt.size())
        gt_loss = torch.mean(gt_prob[mask > 0.5] ** 2) / 2

        out_mask = mask.unsqueeze(1).repeat(1, ndsamples-1, 1, 1) > 0.5

        outputs = {"occ_output": latent_out, "occ_target": latent_target,
                   "occ_mask": out_mask, 'scene_embedding': embeddings}

        loss, _ = model_loss(outputs, depth_gt_ms, mask_ms)
        loss = loss + gt_loss
    else:
        casmvs_outputs = casmvsnet(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        depth_casmvsnet, conf = casmvs_outputs["depth"], casmvs_outputs["photometric_confidence"]
        depth_casmvsnet = F.interpolate(depth_casmvsnet.unsqueeze(1), [height, width], mode='nearest').squeeze(1)
        conf = F.interpolate(conf.unsqueeze(1), [height, width], mode='nearest').squeeze(1)
        mask = conf > 0.8

        inp_latent_net["uv"] = uv.repeat(img_feat.size(0), ndsamples-1, 1)
        inp_latent_net["depth"] = depth_casmvsnet
        latent_out, latent_target, embeddings = model.get_occ_loss(inp_latent_net, mask=mask,
                                                                   ndsamples=ndsamples, trans_vec=sample_cuda["trans_norm"])
        out_mask = mask.unsqueeze(1).repeat(1, ndsamples-1, 1, 1)

        outputs = {"occ_output": latent_out, "occ_target": latent_target,
                   "occ_mask": out_mask, 'scene_embedding': embeddings}
        loss, _ = model_loss(outputs, depth_casmvsnet, mask)

        inp_latent_net["uv"] = uv.repeat(img_feat.size(0), 1, 1)
        inp_latent_net["depth"] = depth_casmvsnet.unsqueeze(1)
        gt_prob, _ = model(inp_latent_net, ndsamples=1, trans_vec=sample_cuda["trans_norm"])
        gt_prob = gt_prob.squeeze(-1).view(depth_casmvsnet.size())
        gt_loss = torch.mean(gt_prob[mask] ** 2) / 2
        loss = loss + gt_loss #+ torch.mean(embeddings ** 2)

    loss.backward()
    optimizer.step()

    return loss.item(), gt_loss.item(), gt_prob[mask > 0.5].max().item()


@make_nograd_func
def test_sample_depth(model, model_loss, sample_cuda, is_begin, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    # sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage1"]
    mask = mask_ms["stage1"]

    height, width = depth_gt.shape[1], depth_gt.shape[2]
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_gt.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth_gt.device)])
    y, x = y.contiguous().view(-1), x.contiguous().view(-1)
    uv = torch.stack([x, y], dim=-1)
    uv = uv.unsqueeze(0)

    ndsamples = 1

    proj_matrices = sample_cuda["proj_matrices"]["stage1"]
    proj_matrices = torch.unbind(proj_matrices, 1)
    refs, srcs = proj_matrices[0], proj_matrices[1:]
    intrinsics, extrinsics = refs[:, 1, :3, :3], refs[:, 0, :4, :4]
    img_feat = F.interpolate(sample_cuda["imgs"][:, 0], [height, width], mode='bilinear', align_corners=False)

    inp_latent_net = {"scene_idx": sample_cuda["scene_idx"], "pose": torch.inverse(extrinsics),
                      "depth": depth_gt, "intrinsics": intrinsics,
                      "uv": uv.repeat(img_feat.size(0), ndsamples, 1),
                      "img_feature": img_feat}

    """if ndsamples > 1:
        depth_interval = np.random.choice(np.arange(10, 100, 5)) #randint(5, 10)
        depth_samples = get_depth_range_samples(cur_depth=depth_gt,
                                                ndepth=ndsamples, depth_inteval_pixel=depth_interval,
                                                dtype=depth_gt.dtype, device=depth_gt.device,
                                                shape=[depth_gt.size(0), depth_gt.size(1), depth_gt.size(2)])
        inp_latent_net["depth"] = depth_samples
    else:
        inp_latent_net["depth"] = inp_latent_net["depth"].unsqueeze(1)

    if mask is not None:
        inp_latent_net["depth"] = inp_latent_net["depth"] * mask.unsqueeze(1).float()

    latent_out, embeddings = model(inp_latent_net, ndsamples=ndsamples)
    latent_out = latent_out.squeeze(-1).view(depth_gt.size(0), ndsamples, depth_gt.size(1), depth_gt.size(2))
    latent_target = torch.zeros((depth_gt.size(0), ndsamples, depth_gt.size(1), depth_gt.size(2)),
                                device=depth_gt.device)
    latent_target[:, (ndsamples - 1) // 2, :, :] = 1.0"""

    latent_out, latent_target, embeddings = model.get_occ_loss(inp_latent_net, mask=mask, ndsamples=ndsamples, is_training=False)
    out_mask = mask.unsqueeze(1).repeat(1, ndsamples, 1, 1) > 0.5

    outputs = {"occ_output": latent_out, "occ_target": latent_target,
               "occ_mask": out_mask, 'scene_embedding': embeddings}

    loss, _ = model_loss(outputs, depth_gt_ms, mask_ms)

    return loss.item(), latent_out[out_mask].min().item(), latent_out[out_mask].mean().item()


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
    model = LatentScene(3, 1, 128, 128, 128, 7)
    model.to(device)
    model_loss = scene_representation_loss

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
        model.load_state_dict(state_dict['model'])
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
            if num_gpus > 1:
                model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale,
                               shuffle=True, seq_size=49, batch_size=args.batch_size, depth_scale=args.depth_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "val", 3, args.numdepth, args.interval_scale,
                              shuffle=False, seq_size=49, batch_size=1, depth_scale=args.depth_scale)

    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    TrainImgLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                num_workers=4, pin_memory=args.pin_m)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    TestImgLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=test_sampler, num_workers=2,
                               pin_memory=args.pin_m)

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError
