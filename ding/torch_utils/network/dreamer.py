import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from ding.torch_utils import MLP
from ding.rl_utils import symlog, inv_symlog


class Conv2dSame(torch.nn.Conv2d):
    """
    Overview:
         Conv2dSame Network for dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def calc_same_pad(self, i, k, s, d):
        """
        Overview:
            Calculate the same padding size.
        Arguments:
            - i (:obj:`int`): Input size.
            - k (:obj:`int`): Kernel size.
            - s (:obj:`int`): Stride size.
            - d (:obj:`int`): Dilation size.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        """
        Overview:
            compute the forward of Conv2dSame.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class DreamerLayerNorm(nn.Module):
    """
    Overview:
         DreamerLayerNorm Network for dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, ch, eps=1e-03):
        """
        Overview:
            Init the DreamerLayerNorm class.
        Arguments:
            - ch (:obj:`int`): Input channel.
            - eps (:obj:`float`): Epsilon.
        """

        super(DreamerLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        """
        Overview:
            compute the forward of DreamerLayerNorm.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DenseHead(nn.Module):
    """
    Overview:
       DenseHead Network for value head, reward head, and discount head of dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        inp_dim,
        shape,  # (255,)
        layer_num,
        units,  # 512
        act='SiLU',
        norm='LN',
        dist='normal',
        std=1.0,
        outscale=1.0,
        device='cpu',
    ):
        """
        Overview:
            Init the DenseHead class.
        Arguments:
            - inp_dim (:obj:`int`): Input dimension.
            - shape (:obj:`tuple`): Output shape.
            - layer_num (:obj:`int`): Number of layers.
            - units (:obj:`int`): Number of units.
            - act (:obj:`str`): Activation function.
            - norm (:obj:`str`): Normalization function.
            - dist (:obj:`str`): Distribution function.
            - std (:obj:`float`): Standard deviation.
            - outscale (:obj:`float`): Output scale.
            - device (:obj:`str`): Device.
        """

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
        self._device = device

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

    def forward(self, features):
        """
        Overview:
            compute the forward of DenseHead.
        Arguments:
            - features (:obj:`torch.Tensor`): Input tensor.
        """

        x = features
        out = self.mlp(x)  # (batch, time, _units=512)
        mean = self.mean_layer(out)  # (batch, time, 255)
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        if self._dist == "normal":
            return ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), len(self._shape)))
        elif self._dist == "huber":
            return ContDist(torchd.independent.Independent(UnnormalizedHuber(mean, std, 1.0), len(self._shape)))
        elif self._dist == "binary":
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(self._shape)))
        elif self._dist == "twohot_symlog":
            return TwoHotDistSymlog(logits=mean, low=-1., high=1., device=self._device)
        raise NotImplementedError(self._dist)


class ActionHead(nn.Module):
    """
    Overview:
       ActionHead Network for action head of dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        """
        Overview:
            Initialize the ActionHead class.
        Arguments:
            - inp_dim (:obj:`int`): Input dimension.
            - size (:obj:`int`): Output size.
            - layers (:obj:`int`): Number of layers.
            - units (:obj:`int`): Number of units.
            - act (:obj:`str`): Activation function.
            - norm (:obj:`str`): Normalization function.
            - dist (:obj:`str`): Distribution function.
            - init_std (:obj:`float`): Initial standard deviation.
            - min_std (:obj:`float`): Minimum standard deviation.
            - max_std (:obj:`float`): Maximum standard deviation.
            - temp (:obj:`float`): Temperature.
            - outscale (:obj:`float`): Output scale.
            - unimix_ratio (:obj:`float`): Unimix ratio.
        """
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = getattr(torch.nn, act)
        self._norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(self._norm(self._units, eps=1e-03))
            pre_layers.append(self._act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

    def forward(self, features):
        """
        Overview:
            compute the forward of ActionHead.
        Arguments:
            - features (:obj:`torch.Tensor`): Input tensor.
        """

        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class SampleDist:
    """
    Overview:
       A kind of sample Dist for ActionHead of dreamerv3.
    Interfaces:
        ``__init__``, ``mean``, ``mode``, ``entropy``
    """

    def __init__(self, dist, samples=100):
        """
        Overview:
            Initialize the SampleDist class.
        Arguments:
            - dist (:obj:`torch.Tensor`): Distribution.
            - samples (:obj:`int`): Number of samples.
        """

        self._dist = dist
        self._samples = samples

    def mean(self):
        """
        Overview:
            Calculate the mean of the distribution.
        """

        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        """
        Overview:
            Calculate the entropy of the distribution.
        """

        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """
    Overview:
       A kind of onehot Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``mode``, ``sample``
    """

    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        """
        Overview:
            Initialize the OneHotDist class.
        Arguments:
            - logits (:obj:`torch.Tensor`): Logits.
            - probs (:obj:`torch.Tensor`): Probabilities.
            - unimix_ratio (:obj:`float`): Unimix ratio.
        """

        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        """
        Overview:
            Sample from the distribution.
        Arguments:
            - sample_shape (:obj:`tuple`): Sample shape.
            - seed (:obj:`int`): Seed.
        """

        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class TwoHotDistSymlog:
    """
    Overview:
       A kind of twohotsymlog Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``mode``, ``mean``, ``log_prob``, ``log_prob_target``
    """

    def __init__(self, logits=None, low=-20.0, high=20.0, device='cpu'):
        """
        Overview:
            Initialize the TwoHotDistSymlog class.
        Arguments:
            - logits (:obj:`torch.Tensor`): Logits.
            - low (:obj:`float`): Low.
            - high (:obj:`float`): High.
            - device (:obj:`str`): Device.
        """

        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255

    def mean(self):
        """
        Overview:
            Calculate the mean of the distribution.
        """

        _mean = self.probs * self.buckets
        return inv_symlog(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        _mode = self.probs * self.buckets
        return inv_symlog(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        """
        Overview:
            Calculate the log probability of the distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        x = symlog(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum((self.buckets > x[..., None]).to(torch.int32), dim=-1)
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = (below == above)

        dist_to_below = torch.where(equal, torch.tensor(1).to(x), torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, torch.tensor(1).to(x), torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None] +
            F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        """
        Overview:
            Calculate the log probability of the target.
        Arguments:
            - target (:obj:`torch.Tensor`): Target tensor.
        """

        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class SymlogDist:
    """
    Overview:
       A kind of Symlog Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``entropy``, ``mode``, ``mean``, ``log_prob``
    """

    def __init__(self, mode, dist='mse', aggregation='sum', tol=1e-8, dim_to_reduce=[-1, -2, -3]):
        """
        Overview:
            Initialize the SymlogDist class.
        Arguments:
            - mode (:obj:`torch.Tensor`): Mode.
            - dist (:obj:`str`): Distribution function.
            - aggregation (:obj:`str`): Aggregation function.
            - tol (:obj:`float`): Tolerance.
            - dim_to_reduce (:obj:`list`): Dimension to reduce.
        """
        self._mode = mode
        self._dist = dist
        self._aggregation = aggregation
        self._tol = tol
        self._dim_to_reduce = dim_to_reduce

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        return inv_symlog(self._mode)

    def mean(self):
        """
        Overview:
            Calculate the mean of the distribution.
        """

        return inv_symlog(self._mode)

    def log_prob(self, value):
        """
        Overview:
            Calculate the log probability of the distribution.
        Arguments:
            - value (:obj:`torch.Tensor`): Input tensor.
        """

        assert self._mode.shape == value.shape
        if self._dist == 'mse':
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == 'abs':
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._aggregation == 'mean':
            loss = distance.mean(self._dim_to_reduce)
        elif self._aggregation == 'sum':
            loss = distance.sum(self._dim_to_reduce)
        else:
            raise NotImplementedError(self._aggregation)
        return -loss


class ContDist:
    """
    Overview:
       A kind of ordinary Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``entropy``, ``mode``, ``sample``, ``log_prob``
    """

    def __init__(self, dist=None):
        """
        Overview:
            Initialize the ContDist class.
        Arguments:
            - dist (:obj:`torch.Tensor`): Distribution.
        """

        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        """
        Overview:
            Get attribute.
        Arguments:
            - name (:obj:`str`): Attribute name.
        """

        return getattr(self._dist, name)

    def entropy(self):
        """
        Overview:
            Calculate the entropy of the distribution.
        """

        return self._dist.entropy()

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        return self._dist.mean

    def sample(self, sample_shape=()):
        """
        Overview:
            Sample from the distribution.
        Arguments:
            - sample_shape (:obj:`tuple`): Sample shape.
        """

        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    """
    Overview:
       A kind of Bernoulli Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``entropy``, ``mode``, ``sample``, ``log_prob``
    """

    def __init__(self, dist=None):
        """
        Overview:
            Initialize the Bernoulli distribution.
        Arguments:
            - dist (:obj:`torch.Tensor`): Distribution.
        """

        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        """
        Overview:
            Get attribute.
        Arguments:
            - name (:obj:`str`): Attribute name.
        """

        return getattr(self._dist, name)

    def entropy(self):
        """
        Overview:
            Calculate the entropy of the distribution.
        """
        return self._dist.entropy()

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        """
        Overview:
            Sample from the distribution.
        Arguments:
            - sample_shape (:obj:`tuple`): Sample shape.
        """

        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        """
        Overview:
            Calculate the log probability of the distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class UnnormalizedHuber(torchd.normal.Normal):
    """
    Overview:
       A kind of UnnormalizedHuber Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``mode``, ``log_prob``
    """

    def __init__(self, loc, scale, threshold=1, **kwargs):
        """
        Overview:
            Initialize the UnnormalizedHuber class.
        Arguments:
            - loc (:obj:`torch.Tensor`): Location.
            - scale (:obj:`torch.Tensor`): Scale.
            - threshold (:obj:`float`): Threshold.
        """
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        """
        Overview:
            Calculate the log probability of the distribution.
        Arguments:
            - event (:obj:`torch.Tensor`): Event.
        """

        return -(torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2) - self._threshold)

    def mode(self):
        """
        Overview:
            Calculate the mode of the distribution.
        """

        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    """
    Overview:
       A kind of SafeTruncatedNormal Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``sample``
    """

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        """
        Overview:
            Initialize the SafeTruncatedNormal class.
        Arguments:
            - loc (:obj:`torch.Tensor`): Location.
            - scale (:obj:`torch.Tensor`): Scale.
            - low (:obj:`float`): Low.
            - high (:obj:`float`): High.
            - clip (:obj:`float`): Clip.
            - mult (:obj:`float`): Mult.
        """

        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        """
        Overview:
            Sample from the distribution.
        Arguments:
            - sample_shape (:obj:`tuple`): Sample shape.
        """

        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    """
    Overview:
       A kind of TanhBijector Dist for dreamerv3.
    Interfaces:
        ``__init__``, ``_forward``, ``_inverse``, ``_forward_log_det_jacobian``
    """

    def __init__(self, validate_args=False, name='tanh'):
        """
        Overview:
            Initialize the TanhBijector class.
        Arguments:
            - validate_args (:obj:`bool`): Validate arguments.
            - name (:obj:`str`): Name.
        """

        super().__init__()

    def _forward(self, x):
        """
        Overview:
            Calculate the forward of the distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        return torch.tanh(x)

    def _inverse(self, y):
        """
        Overview:
            Calculate the inverse of the distribution.
        Arguments:
            - y (:obj:`torch.Tensor`): Input tensor.
        """

        y = torch.where((torch.abs(y) <= 1.), torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        """
        Overview:
            Calculate the forward log det jacobian of the distribution.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan(fn, inputs, start):
    """
    Overview:
         Static scan function.
    Arguments:
        - fn (:obj:`function`): Function.
        - inputs (:obj:`tuple`): Inputs.
        - start (:obj:`torch.Tensor`): Start tensor.
    """

    last = start  # {logit, stoch, deter:[batch_size, self._deter]}
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)  # inputs:(action:(time, batch, 6), embed:(time, batch, 4096))
        last = fn(last, *inp(index))  # post, prior
        if flag:
            if isinstance(last, dict):
                outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
            else:
                outputs = []
                for _last in last:
                    if isinstance(_last, dict):
                        outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if isinstance(last, dict):
                for key in last.keys():
                    outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat([outputs[j][key], last[j][key].unsqueeze(0)], dim=0)
                    else:
                        outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
    if isinstance(last, dict):
        outputs = [outputs]
    return outputs


def weight_init(m):
    """
    Overview:
       weight_init for Linear, Conv2d, ConvTranspose2d, and LayerNorm.
    Arguments:
        - m (:obj:`torch.nn`): Module.
    """

    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    """
    Overview:
       weight_init for Linear and LayerNorm.
    Arguments:
        - given_scale (:obj:`float`): Given scale.
    """

    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    return f
