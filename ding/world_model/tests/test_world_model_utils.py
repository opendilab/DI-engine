import pytest
from easydict import EasyDict
import torch
from torch import distributions as torchd
from itertools import product
from ding.world_model.utils import get_rollout_length_scheduler, SampleDist, OneHotDist, TwoHotDistSymlog, SymlogDist, ContDist, Bernoulli, UnnormalizedHuber, weight_init, uniform_weight_init


@pytest.mark.unittest
def test_get_rollout_length_scheduler():
    fake_cfg = EasyDict(
        type='linear',
        rollout_start_step=20000,
        rollout_end_step=150000,
        rollout_length_min=1,
        rollout_length_max=25,
    )
    scheduler = get_rollout_length_scheduler(fake_cfg)
    assert scheduler(0) == 1
    assert scheduler(19999) == 1
    assert scheduler(150000) == 25
    assert scheduler(1500000) == 25


B, time = 16, 64
mean = torch.randn(B, time, 255)
std = 1.0
a = torch.randn(B, time, 1)  #  or torch.randn(B, time, 255)
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
