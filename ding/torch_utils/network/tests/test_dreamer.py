import pytest
from easydict import EasyDict
import torch
from torch import distributions as torchd
from itertools import product
from ding.torch_utils.network.dreamer import DenseHead, SampleDist, OneHotDist, TwoHotDistSymlog, \
    SymlogDist, ContDist, Bernoulli, UnnormalizedHuber, weight_init, uniform_weight_init

# arguments
shape = [255, (255, ), ()]
# to do
# dist = ['normal', 'huber', 'binary', 'twohot_symlog']
dist = ['twohot_symlog']
args = list(product(*[shape, dist]))


@pytest.mark.unittest
@pytest.mark.parametrize('shape, dist', args)
def test_DenseHead(shape, dist):
    in_dim, layer_num, units, B, time = 1536, 2, 512, 16, 64
    head = DenseHead(in_dim, shape, layer_num, units, dist=dist)
    x = torch.randn(B, time, in_dim)
    a = torch.randn(B, time, 1)
    y = head(x)
    assert y.mode().shape == (B, time, 1)
    assert y.log_prob(a).shape == (B, time)


B, time = 16, 64
mean = torch.randn(B, time, 255)
std = 1.0
a = torch.randn(B, time, 1)  # or torch.randn(B, time, 255)
sample_shape = torch.Size([])


@pytest.mark.unittest
def test_ContDist():
    dist_origin = torchd.normal.Normal(mean, std)
    dist = torchd.independent.Independent(dist_origin, 1)
    dist_new = ContDist(dist)
    assert dist_new.mode().shape == (B, time, 255)
    assert dist_new.log_prob(a).shape == (B, time)
    assert dist_origin.log_prob(a).shape == (B, time, 255)
    assert dist_new.sample().shape == (B, time, 255)


@pytest.mark.unittest
def test_UnnormalizedHuber():
    dist_origin = UnnormalizedHuber(mean, std)
    dist = torchd.independent.Independent(dist_origin, 1)
    dist_new = ContDist(dist)
    assert dist_new.mode().shape == (B, time, 255)
    assert dist_new.log_prob(a).shape == (B, time)
    assert dist_origin.log_prob(a).shape == (B, time, 255)
    assert dist_new.sample().shape == (B, time, 255)


@pytest.mark.unittest
def test_Bernoulli():
    dist_origin = torchd.bernoulli.Bernoulli(logits=mean)
    dist = torchd.independent.Independent(dist_origin, 1)
    dist_new = Bernoulli(dist)
    assert dist_new.mode().shape == (B, time, 255)
    assert dist_new.log_prob(a).shape == (B, time, 255)
    # to do
    # assert dist_new.sample().shape == (B, time, 255)


@pytest.mark.unittest
def test_TwoHotDistSymlog():
    dist = TwoHotDistSymlog(logits=mean)
    assert dist.mode().shape == (B, time, 1)
    assert dist.log_prob(a).shape == (B, time)
