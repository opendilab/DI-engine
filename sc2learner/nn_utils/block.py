import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_module import conv2d_block


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=nn.ReLU(), norm_type='BN'):
        super(ResBlock, self).__init__()
        assert(stride == 1)
        self.act = activation
        self.conv1 = conv2d_block(in_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=norm_type)
        self.conv2 = conv2d_block(out_channels, out_channels, kernel_size, stride,
                                  padding, activation=self.act, norm_type=norm_type)
        self.conv3 = conv2d_block(out_channels, out_channels, 1, 1, 0, activation=None, norm_type=norm_type)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(x + residual)
        return x
