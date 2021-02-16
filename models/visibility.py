import torch
from torch import nn
import torch.nn.functional as F

from .prior_net import UNet
from models.utils.warping import homo_warping_2D


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def set_bn(model, momentum):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def init_uniform(module):
    if module.weight is not None:
        # nn.init.kaiming_uniform_(module.weight)
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def set_eps(model, eps):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eps = eps


class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes.
    optionally followed by batch normalization and relu activation
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.
       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu
       Notes:
           Default momentum for batch normalization is set to be 0.01,
       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.
       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu
       Notes:
           Default momentum for batch normalization is set to be 0.01,
       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
        if self.bn is not None:
            init_bn(self.bn)


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True,
                 bn_momentum=0.1):
        """Multilayer perceptron shared on resolution (1D or 2D)
        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization
        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv2d
        else:
            raise ValueError()

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1,
                                   relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class VisNet(nn.Module):
    def __init__(self, feature_channels=8, depth_channels=0,
                 occ_shared_channels=(128, 128, 128), occ_global_channels=(64, 16, 4)):
        super(VisNet, self).__init__()
        # self.occ_shared_mlp = SharedMLP(2 * feature_channels + depth_channels, occ_shared_channels, ndim=2, bn=True)
        # self.feat_mlp = nn.ModuleList([SharedMLP(2 * c + depth_channels, (occ_shared_channels[0],), ndim=2)
        #                                for c in feature_channels])

        # self.occ_shared_mlp = SharedMLP(occ_shared_channels[0], occ_shared_channels[1:], ndim=2)
        # self.occ_global_mlp = SharedMLP(occ_shared_channels[-1], occ_global_channels, bn=True)

        self.occ_unet = UNet(2 * feature_channels + depth_channels, 32, 32, 3, batchnorms=True)
        self.occ_pred = nn.Sequential(nn.Conv2d(32, 1, 1),
                                      nn.Sigmoid())

    def forward(self, features, proj_matrices, prior_depths, depth_min=None, depth_max=None):

        assert len(features) == proj_matrices.size(1), "Different number of images and projection matrices"
        assert len(features)-1 == prior_depths.size(1), "Wrong number of views in source prior depths"
        num_views = len(features)

        # normalize depth
        batch_size, num_channels, height, width = features[0].size()
        depth_min, depth_max = depth_min.view(-1, 1, 1, 1, 1), depth_max.view(-1, 1, 1, 1, 1)

        all_view_features = torch.stack(features, dim=1)  # .detach() # [B, N, C, H, W]
        all_depths = torch.cat((torch.zeros_like(prior_depths[:, [0], ...]), prior_depths), dim=1)  # [B, N, 1, H, W]
        warped_prior_depths, warped_prior_feats = homo_warping_2D(all_depths, all_view_features, proj_matrices)  # [B, N-1, C, H, W]
        warped_prior_depths = (warped_prior_depths - depth_min) / (depth_max - depth_min)
        warped_prior_depths = warped_prior_depths.view(-1, 1, height, width)
        warped_prior_feats = warped_prior_feats.contiguous().view(-1, num_channels, height, width)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature = features[0]

        # this is for visibility prediction
        ref_feat = ref_feature.repeat(num_views - 1, 1, 1, 1)
        vis_inputs = torch.cat((ref_feat, warped_prior_feats, warped_prior_depths), dim=1)
        vis_maps = self.occ_unet(vis_inputs)  # [B*(N-1), 32, H, W]
        vis_maps = self.occ_pred(vis_maps)  # [B*(N-1), 1, H, W]
        vis_maps = vis_maps.view(batch_size, -1, 1, height, width)  # [B, N-1, 1, H, W]
        return vis_maps
