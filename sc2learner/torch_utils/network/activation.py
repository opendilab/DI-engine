"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build activation: you can use build_activation to build relu or glu
"""
import torch
import torch.nn as nn


class GLU(nn.Module):
    r"""
    Overview:
        a glu nn module

        Note:
            For beginner, you can reference <https://www.jianshu.com/p/d4793635a4c4> to learn more about activation functions.

    Interface:
        __init__, forward
    """
    def __init__(self, input_dim, output_dim, context_dim, input_type='fc'):
        r"""
        Overview:
            Init glu

        Arguments:
            - input_dim (:obj:`int`): the input dimension
            - output_dim (:obj:`int`): the output dimension
            - context_dim (:obj:`int`): the context dimension
            - input_type (:obj:`str`): the type of input, now support ['fc', 'conv2d']
        """
        super(GLU, self).__init__()
        assert (input_type in ['fc', 'conv2d'])
        if input_type == 'fc':
            self.layer1 = nn.Linear(context_dim, input_dim)
            self.layer2 = nn.Linear(input_dim, output_dim)
        elif input_type == 'conv2d':
            self.layer1 = nn.Conv2d(context_dim, input_dim, 1, 1, 0)
            self.layer2 = nn.Conv2d(input_dim, output_dim, 1, 1, 0)

    def forward(self, x, context):
        r"""
        Overview:
            return glu computed tensor

        Arguments:
            - x (:obj:`tensor`) : the input tensor
            - context (:obj:`tensor`) : the context tensor

        Returns:
            - x (:obj:`tensor`): the computed tensor
        """
        gate = self.layer1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.layer2(x)
        return x


def build_activation(activation):
    r"""
    Overview:
        return the activation module match the given activation descripion

    Arguments:
        - actvation (:obj:`str`): the type of activation module needed, now support ['relu', 'glu']

    Returns:
        - act_func (:obj:`torch.nn.module`): the corresponding activation module
    """
    act_func = {'relu': nn.ReLU(inplace=False), 'glu': GLU}
    if activation in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))
