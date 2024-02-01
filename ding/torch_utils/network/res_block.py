from typing import Union

import torch
import torch.nn as nn

from .nn_module import conv2d_block, fc_block


class ResBlock(nn.Module):
    """
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

    .. note::
        You can refer to `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ for more \
        details.

    Interfaces:
        ``__init__``, ``forward``
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
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): Type of the normalization, default set to 'BN'(Batch Normalization), \
                supports ['BN', 'LN', 'IN', 'GN', 'SyncBN', None].
            - res_type (:obj:`str`): Type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): Whether to add a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None, \
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
        """
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
    """
    Overview:
        Residual Block with 2 fully connected layers.
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self, in_channels: int, activation: nn.Module = nn.ReLU(), norm_type: str = 'BN', dropout: float = None
    ):
        """
        Overview:
            Init the fully connected layer residual block.
        Arguments:
            - in_channels (:obj:`int`): The number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): The type of the normalization, default set to 'BN'.
            - dropout (:obj:`float`): The dropout rate, default set to None.
        """
        super(ResFCBlock, self).__init__()
        self.act = activation
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm_type=norm_type)
        self.fc2 = fc_block(in_channels, in_channels, activation=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the output of the redisual block.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x + identity)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TemporalSpatialResBlock(nn.Module):
    """
    Overview:
        Residual Block using MLP layers for both temporal and spatial input.
        t → time_mlp  →  h1 → dense2 → h2 → out
                       ↗+                ↗+
        x →  dense1 →                  ↗
          ↘                          ↗
            → modify_x →   →   →   →
    """

    def __init__(self, input_dim, output_dim, t_dim=128, activation=torch.nn.SiLU()):
        """
        Overview:
            Init the temporal spatial residual block.
        Arguments:
            - input_dim (:obj:`int`): The number of channels in the input tensor.
            - output_dim (:obj:`int`): The number of channels in the output tensor.
            - t_dim (:obj:`int`): The dimension of the temporal input.
            - activation (:obj:`nn.Module`): The optional activation function.
        """
        super().__init__()
        # temporal input is the embedding of time, which is a Gaussian Fourier Feature tensor
        self.time_mlp = nn.Sequential(
            activation,
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim), activation)
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim), activation)
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, t) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
            - t (:obj:`torch.Tensor`): The temporal input tensor.
        """
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)
