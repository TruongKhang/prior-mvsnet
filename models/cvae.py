import torch.nn as nn


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, padding=True):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = []
        output_dim = None
        for i in range(len(self.num_filters[:-1])):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))
        self.last_downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.latent_mu = nn.Sequential(nn.Conv2d(output_dim, self.num_filters[-1]//2, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.num_filters[-1]//2, self.num_filters[-1]//2, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.latent_sigma = nn.Sequential(nn.Conv2d(output_dim, self.num_filters[-1] // 2, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(self.num_filters[-1] // 2, self.num_filters[-1] // 2, kernel_size=3,
                                                    padding=1),
                                          nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)
        self.latent_mu.apply(init_weights)
        self.latent_sigma.apply(init_weights)

    def forward(self, inputs):
        # print(input.size(), self.input_channels)
        output = self.layers(inputs)
        output = self.last_downsample(output)
        mu = self.latent_mu(output)
        sigma = self.latent_sigma(output)
        return mu, sigma