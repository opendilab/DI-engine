import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from ding.world_model.utils import weight_init, uniform_weight_init, ContDist, Bernoulli, TwoHotDistSymlog, UnnormalizedHuber
from ding.torch_utils import MLP, fc_block


class DenseHead(nn.Module):

    def __init__(
        self,
        inp_dim,  # config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        shape,  # (255,)
        layer_num,
        units,  # 512
        act='SiLU',
        norm='LN',
        dist='normal',
        std=1.0,
        outscale=1.0,
    ):
        super(DenseHead, self).__init__()
        self._shape = (shape, ) if isinstance(shape, int) else shape
        if len(self._shape) == 0:
            self._shape = (1, )
        self._layer_num = layer_num
        self._units = units
        self._act = getattr(torch.nn, act)()
        self._norm = norm
        self._dist = dist
        self._std = std

        self.mlp = MLP(
            inp_dim,
            self._units,
            self._units,
            self._layer_num,
            layer_fn=nn.Linear,
            activation=self._act,
            norm_type=self._norm
        )
        self.mlp.apply(weight_init)

        self.mean_layer = nn.Linear(self._units, np.prod(self._shape))
        self.mean_layer.apply(uniform_weight_init(outscale))

        if self._std == "learned":
            self.std_layer = nn.Linear(self._units, np.prod(self._shape))
            self.std_layer.apply(uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        out = self.mlp(x)  # (batch, time, _units=512)
        mean = self.mean_layer(out)  # (batch, time, 255)
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        if self._dist == "normal":
            return ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
        if self._dist == "huber":
            return ContDist(torchd.independent.Independent(UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
        if self._dist == "binary":
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
        if self._dist == "twohot_symlog":
            return TwoHotDistSymlog(logits=mean)
        raise NotImplementedError(self._dist)
