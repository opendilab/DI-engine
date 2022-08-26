import torch.nn as nn
import torch

from .nn_module import conv2d_block, fc_block, build_normalization


class ResBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2D convolution layers, including 2 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+
    Interfaces:
        forward
    '''

    def __init__(
            self,
            in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN',
            res_type: str = 'basic'
    ) -> None:
        r"""
        Overview:
            Init the Residual Block
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to 'BN'(Batch Normalization), \
                supports ['BN', 'IN', 'SyncBN', None].
            - res_type (:obj:`str`): type of residual block, supports ['basic', 'bottleneck']
        """
        super(ResBlock, self).__init__()
        self.act = activation
        assert res_type in ['basic',
                            'bottleneck'], 'residual type only support basic and bottleneck, not:{}'.format(res_type)
        self.res_type = res_type
        if self.res_type == 'basic':
            self.conv1 = conv2d_block(in_channels, in_channels, 3, 1, 1, activation=self.act, norm_type=norm_type)
            self.conv2 = conv2d_block(in_channels, in_channels, 3, 1, 1, activation=None, norm_type=norm_type)
        elif self.res_type == 'bottleneck':
            self.conv1 = conv2d_block(in_channels, in_channels, 1, 1, 0, activation=self.act, norm_type=norm_type)
            self.conv2 = conv2d_block(in_channels, in_channels, 3, 1, 1, activation=self.act, norm_type=norm_type)
            self.conv3 = conv2d_block(in_channels, in_channels, 1, 1, 0, activation=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the residual block output
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - x(:obj:`torch.Tensor`): the resblock output tensor
        """
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.res_type == 'bottleneck':
            x = self.conv3(x)
        x = self.act(x + residual)
        return x


class ResFCBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interfaces:
        forward
    '''

    def __init__(
        self, in_channels: int, activation: nn.Module = nn.ReLU(), norm_type: str = 'BN', final_norm: bool = False
    ):
        r"""
        Overview:
            Init the Residual Block
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, default set to 'BN'
            - final_norm (:obj:`bool`): Whether to add norm in final residual output.
        """
        super(ResFCBlock, self).__init__()
        self.act = activation
        self.final_norm = final_norm
        if final_norm:
            self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm_type=None)
            self.fc2 = fc_block(in_channels, in_channels, activation=None, norm_type=None)
            self.norm = build_normalization(norm_type)(in_channels)
        else:
            self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm_type=norm_type)
            self.fc2 = fc_block(in_channels, in_channels, activation=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the redisual block output
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - x (:obj:`torch.Tensor`): the resblock output tensor
        """
        if self.final_norm:
            residual = x
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.norm(x + residual)
            return x
        else:
            residual = x
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.act(x + residual)
            return x


class GatedConvResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU(), norm_type='BN'):
        super(GatedConvResBlock, self).__init__()
        assert (stride == 1), stride
        assert (in_channels == out_channels), '{}/{}'.format(in_channels, out_channels)
        self.act = activation
        self.conv1 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type)
        self.conv2 = conv2d_block(out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type)
        self.gate = nn.Sequential(
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=None),
            conv2d_block(out_channels, out_channels, 1, 1, 0, activation=None, norm_type=None)
        )
        self.update_sp = nn.Parameter(torch.full((1, ), fill_value=0.1))

    def forward(self, x, noise_map):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.tanh(x * torch.sigmoid(self.gate(noise_map))) * self.update_sp
        x = self.act(x + residual)
        return x
