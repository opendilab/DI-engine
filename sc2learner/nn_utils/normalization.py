import torch
import torch.nn as nn
import torch.nn.functional as F
import linklink as link
from sc2learner.utils import get_group


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    def __init__(self, num_features,
                 bn_group_size=None,
                 momentum=0.1,
                 sync_stats=True,
                 var_mode=link.syncbnVarMode_t.L2):

        self.group_size = bn_group_size
        super(GroupSyncBatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            group=get_group(bn_group_size),
            sync_stats=sync_stats,
            var_mode=var_mode,
        )

    def __repr__(self):
        return ('{name}({num_features},'
                ' eps={eps},'
                ' momentum={momentum},'
                ' affine={affine},'
                ' group={group},'
                ' group_size={group_size},'
                ' sync_stats={sync_stats},'
                ' var_mode={var_mode})'.format(name=self.__class__.__name__, **self.__dict__))


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = None
        self.bias = None

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None
        b, c, h, w = x.shape
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshape = x.contiguous().view(1, b*c, h, w)
        output = F.batch_norm(
            x_reshape, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        return output.view(b, c, h, w)


def build_normalization(norm_type, dim=None):
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
