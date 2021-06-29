"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. build normalization: you can use classes in this file to build normalizations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from nervex.utils import get_group, try_import_link

link = try_import_link()


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    """
    Overview:
        Apply Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]-D inputs with additional channel
        dimension) as described in the paper:
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
    Interface:
        __init__, __repr__
    
    .. note::

        you can refer to https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html

    .. tip::

        This class relies on linklink, you can find details at:
        http://spring.sensetime.com/docs/linklink/api/index.html#syncbn
    """

    def __init__(
            self,
            num_features: int,
            bn_group_size: int = None,
            momentum: float = 0.1,
            sync_stats: bool = True,
            var_mode: Any = link.syncbnVarMode_t.L2
    ) -> None:
        """
        Overview:
            Init class GroupSyncBatchNorm
        Arguments:
            - num_features (:obj:`int`): size of input feature, C of (N, C, +)
            - bn_group_size (:obj:`int`): synchronization of stats happen within each process group \
                individually Default behavior is synchronization across the whole world
            - momentum (:obj:`float`): the value used for the running_mean and running_var\
                computation. Can be set to ``None`` for cumulative moving average
            - sync_stats (:obj:`bool`): a boolean value that when set to True, this module will\
                average the running mean and variance among all ranks; and when set to False,\
                the running mean and variance only track statistics among the group. Default: False
            - var_mode (:obj:`Any`): when set to linklink.nn.syncbnVarMode_t.L1, will use L1 norm\
                mentioned in Norm matters: efficient and accurate normalization schemes in deep networks
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
            Return the basic information of the class
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


def build_normalization(norm_type: str, dim: int = None) -> nn.Module:
    r"""
    Overview:
        Build the corresponding normalization module
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'IN', 'SyncBN', 'AdaptiveIN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]
    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function
    
    .. note::
        For beginers, you can refer to <https://zhuanlan.zhihu.com/p/34879333> to learn more about batch normalization.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN', 'SyncBN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN2': nn.InstanceNorm2d,
        'SyncBN2': GroupSyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
