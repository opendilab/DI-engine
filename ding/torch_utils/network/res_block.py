from typing import Optional, Union

import torch
import torch.nn as nn

from .nn_module import conv2d_block, fc_block


class ResBlock(nn.Module):
    r"""
    Overview:
        Residual Block with 2D convolution layers, including 3 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+
            downsample block: used in EfficientZero
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________ 3*3*C ____________________/+
    Interfaces:
        forward
    """

    def __init__(
        self,
        in_channels: int,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = 'BN',
        res_type: str = 'basic',
        bias: bool = True,
        out_channels: Union[int, None] = None,
    ) -> None:
        """
        Overview:
            Init the 2D convolution residual block.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - activation (:obj:`nn.Module`): the optional activation function.
            - norm_type (:obj:`str`): type of the normalization, default set to 'BN'(Batch Normalization), \
                supports ['BN', 'IN', 'SyncBN', None].
            - res_type (:obj:`str`): type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): whether adds a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None,
                which means out_channels = in_channels.
        """
        super(ResBlock, self).__init__()
        self.act = activation
        assert res_type in ['basic', 'bottleneck',
                            'downsample'], 'residual type only support basic and bottleneck, not:{}'.format(res_type)
        self.res_type = res_type
        if out_channels is None:
            out_channels = in_channels
        if self.res_type == 'basic':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'bottleneck':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(
                out_channels, out_channels, 1, 1, 0, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'downsample':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 2, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(in_channels, out_channels, 3, 2, 1, activation=None, norm_type=None, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.res_type == 'bottleneck':
            x = self.conv3(x)
        elif self.res_type == 'downsample':
            identity = self.conv3(identity)
        x = self.act(x + identity)
        return x


class ResFCBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected layers.
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interfaces:
        forward
    '''

    def __init__(self, in_channels: int, activation: nn.Module = nn.ReLU(), norm_type: str = 'BN'):
        r"""
        Overview:
            Init the fully connected layer residual block.
        Arguments:
            - in_channels (:obj:`int`): The number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): The type of the normalization, default set to 'BN'.
        """
        super(ResFCBlock, self).__init__()
        self.act = activation
        self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm_type=norm_type)
        self.fc2 = fc_block(in_channels, in_channels, activation=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x + identity)
        return x
