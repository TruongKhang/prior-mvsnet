import argparse, collections
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

import datasets.data_loaders as module_data
import models.model as module_arch
import models.losses as module_loss
from trainer import Trainer, PriornetTrainer
from utils import WarmupMultiStepLR
from parse_config import ConfigParser


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
        "data_list": config["data_loader"]["args"]["valid_data_list"],
        "mode": "val",
        "num_srcs": 7,
        "num_depths": config["data_loader"]["args"]["num_depths"],
        "interval_scale": config["data_loader"]["args"]["interval_scale"],
        "shuffle": False,
        "num_stages": config["data_loader"]["args"]["num_stages"],
        "batch_size": 1,
        "load_prior": config["data_loader"]["args"]["load_prior"]
    }
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(**init_kwags)

    use_prior_loss = config["trainer"]["use_prior_loss"]
    # use_prior = config["trainer"]["use_prior"]

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])

    if not use_prior_loss:
        print("Training Prior-MVSNet")
        # build models architecture, then print to console
        model = config.init_obj('arch', module_arch)
        # logger.info(model)
        """print('Load pretrained model')
        checkpoint = torch.load('pretrained/full/blendedmvs.ckpt')
        new_state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            new_state_dict[key.replace('module.', '')] = val
        model.load_state_dict(new_state_dict)
        print('Done')"""

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        mvsnet_params = filter(lambda p: p.requires_grad, model.mvsnet_parameters)
        mvsnet_optimizer = config.init_obj('optimizer', torch.optim, mvsnet_params)
        mvsnet_optimizer.add_param_group(
            {'params': filter(lambda p: p.requires_grad, model.refine_network.parameters()),
             'lr': 0.0001})

        milestones = [len(data_loader) * int(epoch_idx) for epoch_idx in
                      config["trainer"]["lrepochs"].split(':')[0].split(',')]
        lr_gamma = 1 / float(config["trainer"]["lrepochs"].split(':')[1])
        mvsnet_lr_sch = WarmupMultiStepLR(mvsnet_optimizer, milestones, gamma=lr_gamma,
                                          warmup_factor=1.0 / 3, warmup_iters=500)
        lr_scheduler = {"mvsnet": mvsnet_lr_sch}
        # optimizer = [mvsnet_optimizer]
        writer = SummaryWriter(config.log_dir)

        trainer = Trainer(model, criterion, mvsnet_optimizer, config=config, data_loader=data_loader,
                          valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, writer=writer)
    else:
        print("Training PriorNet")
        model = config.init_obj('arch', module_arch)
        priornet_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, priornet_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        writer = SummaryWriter(config.log_dir)
        trainer = PriornetTrainer(model, criterion, optimizer, config=config, data_loader=data_loader,
                                  valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, writer=writer)

    trainer.train()


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
