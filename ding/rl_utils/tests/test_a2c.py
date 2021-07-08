import pytest
from itertools import product
import numpy as np
import torch
from ding.rl_utils import a2c_data, a2c_error

random_weight = torch.rand(4) + 1
weight_args = [None, random_weight]


@pytest.mark.unittest
@pytest.mark.parametrize('weight, ', weight_args)
def test_a2c(weight):
    B, N = 4, 32
    logit = torch.randn(B, N).requires_grad_(True)
    action = torch.randint(0, N, size=(B, ))
    value = torch.randn(B).requires_grad_(True)
    adv = torch.rand(B)
    return_ = torch.randn(B) * 2
    data = a2c_data(logit, action, value, adv, return_, weight)
    loss = a2c_error(data)
    assert all([l.shape == tuple() for l in loss])
    assert logit.grad is None
    assert value.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit.grad, torch.Tensor)
    assert isinstance(value.grad, torch.Tensor)
