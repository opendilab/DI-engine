"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build ResBlock: you can use this classes to build residual blocks
"""
import torch.nn as nn

from .nn_module import conv2d_block, fc_block


class ResBlock(nn.Module):
    r'''
    Overview:
        Residual Block with conv2d_block

        Note:
            For beginners, you can reference <https://www.jianshu.com/p/d4793635a4c4>
            and <https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec>
            to learn more about ResBlock and ResNet.

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU(), norm_type='BN'):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            Notes:
                Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>

            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization,
                                      support ['BN', 'IN', 'SyncBN', None]
        """
        super(ResBlock, self).__init__()
        assert (stride == 1)
        assert (in_channels == out_channels)
        self.act = activation
        self.conv1 = conv2d_block(in_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=norm_type)
        self.conv2 = conv2d_block(
            out_channels, out_channels, kernel_size, stride, padding, activation=self.act, norm_type=norm_type
        )
        # using kernal size = 1 to serve as bottleneck
        self.conv3 = conv2d_block(out_channels, out_channels, 1, 1, 0, activation=None, norm_type=norm_type)

    def forward(self, x):
        r"""
        Overview:
            return the redisual block output

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(x + residual)
        return x


class ResFCBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \____________________________________/+
    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_type='BN'):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            Notes:
                you can reference .nn_module.fcblock
                nn.linear <https://pytorch.org/docs/master/generated/torch.nn.Linear.html>

            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock, self).__init__()
        assert (in_channels == out_channels)
        self.act = activation
        self.fc1 = fc_block(in_channels, out_channels, activation=self.act, norm_type=norm_type)
        self.fc2 = fc_block(out_channels, out_channels, activation=None, norm_type=norm_type)

    def forward(self, x):
        r"""
        Overview:
            return  output of  the redisual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x + residual)
        return x
