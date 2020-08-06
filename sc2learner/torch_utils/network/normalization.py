"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build normalization: you can use classes in this file to build normalizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sc2learner.utils import get_group, try_import_link

link = try_import_link()


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    r"""
    Overview:
       Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper 
       Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

        Notes:
            you can reference https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html

    Interface:
        __init__, __repr__
    """
    def __init__(
        self, num_features, bn_group_size=None, momentum=0.1, sync_stats=True, var_mode=link.syncbnVarMode_t.L2
    ):
        #TODO
        r"""
        Overview:
            Init class GroupSyncBatchNorm
        
        Arguments:
            Notes:
                reference https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        """
        self.group_size = bn_group_size
        super(GroupSyncBatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            group=get_group(bn_group_size),
            sync_stats=sync_stats,
            var_mode=var_mode,
        )

    def __repr__(self):
        r"""
        Overview:
            output the basic information of the class

        Returns:
            - ret (:obj:`str`): the basic information and discription of class GroupSyncBatchNorm
        """
        return (
            '{name}({num_features},'
            ' eps={eps},'
            ' momentum={momentum},'
            ' affine={affine},'
            ' group={group},'
            ' group_size={group_size},'
            ' sync_stats={sync_stats},'
            ' var_mode={var_mode})'.format(name=self.__class__.__name__, **self.__dict__)
        )


class AdaptiveInstanceNorm2d(nn.Module):
    r"""
    Overview:
        the Adaptive Instance Normalization with 2 dimensions.
       
        Notes:
            you can reference <https://www.jianshu.com/p/7aeb1b41930b> or read paper <https://arxiv.org/pdf/1703.06868.pdf>
            to learn more about Adaptive Instance Normalization

    Interface:
        __init__, forward
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        r"""
        Overview:
            Init class AdaptiveInstanceNorm2d

        Arguments:
            Notes:
                reference batch_normal of <https://pytorch.org/docs/stable/nn.functional.html>

            - num_featurnes (:obj:`int`): the number of features
        """
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = None
        self.bias = None

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, x):
        r"""
        Overview:
            compute the output of AdaptiveInstanceNorm

        Arguments:
            - x (:obj:`Tensor`): the batch input tensor of AdaIN

        Shapes:
            - x (:obj:`Tensor`): :math:`(B, C, H, W)`, while B is the batch size, 
                C is number of channels , H and W stands for height and width    
        """
        assert self.weight is not None and self.bias is not None
        b, c, h, w = x.shape
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshape = x.contiguous().view(1, b * c, h, w)
        output = F.batch_norm(
            x_reshape, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return output.view(b, c, h, w)


def build_normalization(norm_type, dim=None):
    r"""
    Overview:
        build the corresponding normalization module
    
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'IN', 'SyncBN', 'AdaptiveIN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]

    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN', 'SyncBN']:
            key = norm_type + str(dim)
        else:
            key = norm_type
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN2': nn.InstanceNorm2d,
        'AdaptiveIN': AdaptiveInstanceNorm2d,
        'SyncBN2': GroupSyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
