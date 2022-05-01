import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.distributions as dis

import time
import random

from gym3.types import Real, TensorType

REAL = Real()


def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs, device=dev())


def NormedLinear(*args, scale=1.0, dtype=torch.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """

    out = nn.Linear(*args, **kwargs)

    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out


def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x


class Encoder(nn.Module):
    """
    Takes in seq of observations and outputs sequence of codes
    Encoders can be stateful, meaning that you pass in one observation at a
    time and update the state, which is a separate object. (This object
    doesn't store any state except parameters)
    """

    def __init__(self, obtype, codetype):
        super().__init__()
        self.obtype = obtype
        self.codetype = codetype

    def initial_state(self, batchsize):
        raise NotImplementedError

    def empty_state(self):
        return None

    def stateless_forward(self, obs):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
        returns:
            codes: array or dict, all with preshape (B, T)
        """
        code, _state = self(obs, None, self.empty_state())
        return code

    def forward(self, obs, first, state_in):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
            first: float array shape (B, T)
            state_in: array or dict, all with preshape (B,)
        returns:
            codes: array or dict
            state_out: array or dict
        """
        raise NotImplementedError


class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList([CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)])

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, "pool", True):
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    name = "ImpalaCNN"  # put it here to preserve pickle compat

    def __init__(self, inshape, chans, outsize, scale_ob, nblock, final_relu=True, **kwargs):
        super().__init__()
        self.scale_ob = scale_ob
        c, h, w = inshape
        curshape = (c, h, w)
        s = 1 / math.sqrt(len(chans))  # per stack scale
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, **kwargs)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = NormedLinear(intprod(curshape), outsize, scale=1.4)
        self.outsize = outsize
        self.final_relu = final_relu

    def forward(self, x):
        x = x.to(dtype=torch.float32) / self.scale_ob

        x = sequential(self.stacks, x, diag_name=self.name)

        x = flatten_image(x)
        x = torch.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = torch.relu(x)
        return x


class ImpalaEncoder(Encoder):

    def __init__(self, inshape, outsize=256, chans=(16, 32, 32), scale_ob=255.0, nblock=2, **kwargs):
        codetype = TensorType(eltype=REAL, shape=(outsize, ))
        obtype = TensorType(eltype=REAL, shape=inshape)
        super().__init__(codetype=codetype, obtype=obtype)
        self.cnn = ImpalaCNN(inshape=inshape, chans=chans, scale_ob=scale_ob, nblock=nblock, outsize=outsize, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        return x

    def initial_state(self, batchsize):
        return zeros(batchsize, 0)
