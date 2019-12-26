import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if norm_type in ['BN', 'IN']:
            key = norm_type + str(dim)
        else:
            key = norm_type
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN2': nn.InstanceNorm2d,
        'AdaptiveIN': AdaptiveInstanceNorm2d,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))
