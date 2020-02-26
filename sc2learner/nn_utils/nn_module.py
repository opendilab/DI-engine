import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, kaiming_normal_, orthogonal_
from .normalization import build_normalization


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
    # conv1d + norm + activation
    block = []
    block.append(nn.Conv1d(in_channels, out_channels,
                           kernel_size, stride, padding, dilation, groups))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    else:
        block.append(build_normalization(norm_type, dims=1)(out_channels))
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
    else:
        block.append(build_normalization(norm_type, dim=2)(out_channels))
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
    block.append(nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups
    ))
    weight_init_(block[-1].weight, init_type, activation)
    if norm_type is None:
        pass
    else:
        block.append(build_normalization(norm_type, dim=2)(out_channels))
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
    else:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


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


def one_hot(val, num, num_first=False):
    '''
        val: Tensor[batch_size, *]
        num: int
    '''
    assert(isinstance(val, torch.Tensor))
    assert(len(val.shape) >= 1)
    old_shape = val.shape
    val_reshape = val.reshape(-1, 1)
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    try:
        ret.scatter_(1, val_reshape, 1)
    except RuntimeError:
        print(val_reshape, num, val_reshape.shape)
        raise RuntimeError
    if num_first:
        return ret.reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)


class NearestUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(NearestUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, self.scale_factor, mode='nearest')


class BilinearUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(BilinearUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, self.scale_factor, mode='bilinear', align_corner=False)


def binary_encode(y, max_val):
    assert(max_val > 0)
    x = y.clamp(0, max_val)
    B = x.shape[0]
    L = int(math.log(max_val, 2)) + 1
    binary = []
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    for i in range(L):
        num = math.pow(2, L-i-1)
        bit = torch.where(x >= num, one, zero)
        x -= bit * num
        binary.append(bit)
    return torch.stack(binary, dim=1)
