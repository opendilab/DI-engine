import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, kaiming_normal_, orthogonal_


def weight_init_(weight, init_type="xavier", activation=None):

    def xavier_init(weight, *args):
        xavier_normal_(weight)

    def kaiming_init(weight, activation):
        assert activation is not None
        if hasattr(activation, "negative_slope"):
            kaiming_normal_(weight, a=activation.negative_slope)
        else:
            kaiming_normal_(weight, a=0)

    def orthogonal_init(weight, **kwargs):
        orthogonal_(weight)

    init_type_dict = {"xavier": xavier_init,
                      "kaiming": kaiming_init,
                      "orthogonal": orthogonal_init}
    if init_type in init_type_dict:
        init_type_dict[init_type](weight, activation)
    else:
        raise ValueError("Invalid Value in init type: {}".format(init_type))


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = (x - mean).pow(2).mean(dim=-1, keepdim=True)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1] * (x.dim() - 1) + [-1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = None
        self.bias = None

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None
        b, c, h, w = x.shape
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshape = x.contiguous().view(1, b*c, h, w)
        output = F.batch_norm(
            x_reshape, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        return output.view(b, c, h, w)


def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv1d_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 init_type="xavier",
                 activation=None,
                 norm_type=None):
    # conv1d + bn + activation
    block = []
    block.append(nn.Conv1d(in_channels, out_channels,
                           kernel_size, stride, padding, dilation, groups))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    elif norm_type == 'BN':
        block.append(nn.BatchNorm1d(out_channels))
    elif norm_type == 'LN':
        block.append(LayerNorm(out_channels, affine=True))
    else:
        raise ValueError
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def conv2d_block_bn(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    init_type="xavier",
                    activation=None,
                    use_batchnorm=False):
    # conv2d + bn + activation
    block = []
    block.append(nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, dilation, groups))
    weight_init_(block[-1].weight, init_type, activation)
    if use_batchnorm:
        block.append(nn.BatchNorm2d(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def conv2d_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 init_type="xavier",
                 pad_type='zero',
                 activation=None,
                 norm_type=None):
    # conv2d + norm + activation
    block = []
    if pad_type == 'zero':
        block.append(nn.ZeroPad2d(padding))
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
    elif pad_type == 'replicate':
        block.append(nn.ReplicatePad2d(padding))
    else:
        raise ValueError
    block.append(nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding=0, dilation=dilation, groups=groups))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    elif norm_type == 'BN':
        block.append(nn.BatchNorm2d(out_channels))
    elif norm_type == 'IN':
        block.append(nn.InstanceNorm2d(out_channels, affine=True))
    elif norm_type == 'LN':
        block.append(LayerNorm(out_channels, affine=True))
    elif norm_type == 'AdaptiveIN':
        block.append(AdaptiveInstanceNorm2d(out_channels))
    else:
        raise ValueError
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def deconv2d_block_bn(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      dilation=1,
                      groups=1,
                      init_type="xavier",
                      pad_type='zero',
                      activation=None,
                      use_batchnorm=False):
    # transpose conv2d + bn + activation
    block = []
    if pad_type == 'zero':
        block.append(nn.ZeroPad2d(padding))
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
    elif pad_type == 'replicate':
        block.append(nn.ReplicatePad2d(padding))
    else:
        raise ValueError
    block.append(nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        output_padding=output_padding,
        groups=groups
    ))
    weight_init_(block[-1].weight, init_type, activation)
    if use_batchnorm:
        block.append(nn.BatchNorm2d(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def deconv2d_block(in_channels,
                   out_channels,
                   kernel_size,
                   stride=1,
                   padding=0,
                   output_padding=0,
                   dilation=1,
                   groups=1,
                   init_type="xavier",
                   pad_type='zero',
                   activation=None,
                   norm_type=None):
    # transpose conv2d + norm + activation
    block = []
    if pad_type == 'zero':
        block.append(nn.ZeroPad2d(padding))
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
    elif pad_type == 'replicate':
        block.append(nn.ReplicatePad2d(padding))
    else:
        raise ValueError
    block.append(nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        output_padding=output_padding,
        groups=groups
    ))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    elif norm_type == 'BN':
        block.append(nn.BatchNorm2d(out_channels))
    elif norm_type == 'IN':
        block.append(nn.InstanceNorm2d(out_channels, affine=True))
    elif norm_type == 'LN':
        block.append(LayerNorm(out_channels, affine=True))
    elif norm_type == 'AdaptiveIN':
        block.append(AdaptiveInstanceNorm2d(out_channels))
    else:
        raise ValueError
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def fc_block(in_channels,
             out_channels,
             init_type="xavier",
             activation=None,
             norm_type=None,
             use_dropout=False,
             dropout_probability=0.5):
    block = []
    block.append(nn.Linear(in_channels, out_channels))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    elif norm_type == 'BN':
        block.append(nn.BatchNorm1d(out_channels))
    elif norm_type == 'LN':
        block.append(LayerNorm(out_channels, affine=True))
    else:
        raise ValueError
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1,
                 padding=None, init_type="xavier", activation=nn.ReLU(),
                 norm_type=None, is_bottleneck=False, scaling_factor=1.):
        super(ResidualBlock, self).__init__()
        assert stride in [1, 2]
        if out_channels is None:
            out_channels = in_channels // stride
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = conv2d_block(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         padding=0,
                                         init_type=init_type,
                                         activation=None,
                                         norm_type=norm_type)

        block = []
        if is_bottleneck:
            block.append(conv2d_block(in_channels=in_channels,
                                      out_channels=in_channels // 2,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      init_type=init_type,
                                      activation=activation,
                                      norm_type=norm_type))
            block.append(conv2d_block(in_channels=in_channels // 2,
                                      out_channels=out_channels // 2,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=(kernel_size - 1) // 2,
                                      init_type=init_type,
                                      activation=activation,
                                      norm_type=norm_type))
            block.append(conv2d_block(in_channels=out_channels // 2,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      init_type=init_type,
                                      activation=None,
                                      norm_type=norm_type))
        else:
            if padding is None:
                padding = (kernel_size - 1) // 2
            block.append(conv2d_block(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      init_type=init_type,
                                      activation=activation,
                                      norm_type=norm_type))
            block.append(conv2d_block(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=padding,
                                      init_type=init_type,
                                      activation=None,
                                      norm_type=norm_type))
        self.scaling_factor = scaling_factor
        self.activation = activation
        self.block = sequential_pack(block)

    def forward(self, x):
        return self.activation(self.block(x) +
                               self.scaling_factor * self.shortcut(x))


class ChannelShuffle(nn.Module):
    def __init__(self, group_num):
        super(ChannelShuffle, self).__init__()
        self.group_num = group_num

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.group_num
        assert(c % g == 0)
        x = x.view(b, g, c//g, h, w).permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        return x
