from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.distributions as dis

from ding.torch_utils import ResFCBlock, ResBlock, Flatten
from ding.utils import SequenceType

import math
import time
import random


class ConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [32, 64, 64, 128],
            activation: Optional[nn.Module] = nn.ReLU(),
            kernel_size: SequenceType = [8, 4, 3],
            stride: SequenceType = [4, 2, 1],
            padding: Optional[SequenceType] = None,
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Convolution Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers`` and ``ResBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        if padding is None:
            padding = [0 for _ in range(len(kernel_size))]

        layers = []
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i], padding[i]))
            layers.append(self.act)
            input_size = hidden_size_list[i]
        assert len(set(hidden_size_list[3:-1])) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(3, len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i], activation=self.act, norm_type=norm_type))
        layers.append(Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class FCEncoder(nn.Module):
    r"""
    Overview:
        The ``FCEncoder`` used in models. Used to encoder raw 1-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - res_block (:obj:`bool`): Whether use ``res_block``.
            - activation (:obj:`nn.Module`):
                The type of activation to use in the ``ResFCBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResFCBlock`` for more details
        """
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        if res_block:
            assert len(set(hidden_size_list)) == 1, "Please indicate the same hidden size for res block parts"
            if len(hidden_size_list) == 1:
                self.main = ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type)
            else:
                layers = []
                for i in range(len(hidden_size_list)):
                    layers.append(ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type))
                self.main = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(hidden_size_list) - 1):
                layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
                layers.append(self.act)
            self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x


class StructEncoder(nn.Module):
    # TODO(nyz)
    pass



def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


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


class Encoder(nn.Module):
    """
    Takes in seq of observations and outputs sequence of codes
    Encoders can be stateful, meaning that you pass in one observation at a
    time and update the state, which is a separate object. (This object
    doesn't store any state except parameters)
    """

    def __init__(self):
        super().__init__()

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
        self.stacks = nn.Sequential()
        for outchan in chans:
            stack = CnnDownStack(curshape[0], nblock=nblock, outchan=outchan, scale=s, **kwargs)
            self.stacks.add_module(stack)
            curshape = stack.output_shape(curshape)
        self.dense = NormedLinear(intprod(curshape), outsize, scale=1.4)
        self.outsize = outsize
        self.final_relu = final_relu

    def forward(self, x):
        x = x.to(dtype=torch.float32) / self.scale_ob

        x = self.stacks(x)

        x = Flatten(x)
        x = torch.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = torch.relu(x)
        return x


class ImpalaEncoder(Encoder):

    def __init__(self, inshape, outsize=256, chans=(16, 32, 32), scale_ob=255.0, nblock=2, **kwargs):
        super().__init__()
        self.cnn = ImpalaCNN(inshape=inshape, chans=chans, scale_ob=scale_ob, nblock=nblock, outsize=outsize, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        return x

    def initial_state(self, batchsize):
        return torch.zeros(batchsize, 0)
