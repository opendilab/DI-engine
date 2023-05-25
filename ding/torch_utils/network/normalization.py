from typing import Optional
import torch.nn as nn


def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:
    r"""
    Overview:
        Build the corresponding normalization module
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'LN', 'IN', 'SyncBN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]
    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function

    .. note::
        For beginers, you can refer to <https://zhuanlan.zhihu.com/p/34879333> to learn more about batch normalization.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN', 'SyncBN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN1': nn.InstanceNorm1d,
        'IN2': nn.InstanceNorm2d,
        'SyncBN': nn.SyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
