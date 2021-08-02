import pytest
from itertools import product
import numpy as np
import torch

from ding.rl_utils import ppo_data, ppo_error, ppo_error_continuous
from ding.rl_utils.ppo import shape_fn_ppo

use_value_clip_args = [True, False]
dual_clip_args = [None, 5.0]
random_weight = torch.rand(4) + 1
weight_args = [None, random_weight]
args = [item for item in product(*[use_value_clip_args, dual_clip_args, weight_args])]


@pytest.mark.unittest
def test_shape_fn_ppo():
    data = ppo_data(torch.randn(3, 5, 8), None, None, None, None, None, None, None)
    shape1 = shape_fn_ppo([data], {})
    shape2 = shape_fn_ppo([], {'data': data})
    assert shape1 == shape2 == (3, 5, 8)


@pytest.mark.unittest
@pytest.mark.parametrize('use_value_clip, dual_clip, weight', args)
def test_ppo(use_value_clip, dual_clip, weight):
    B, N = 4, 32
    logit_new = torch.randn(B, N).requires_grad_(True)
    logit_old = logit_new + torch.rand_like(logit_new) * 0.1
    action = torch.randint(0, N, size=(B, ))
    value_new = torch.randn(B).requires_grad_(True)
    value_old = value_new + torch.rand_like(value_new) * 0.1
    adv = torch.rand(B)
    return_ = torch.randn(B) * 2
    data = ppo_data(logit_new, logit_old, action, value_new, value_old, adv, return_, weight)
    loss, info = ppo_error(data, use_value_clip=use_value_clip, dual_clip=dual_clip)
    assert all([l.shape == tuple() for l in loss])
    assert all([np.isscalar(i) for i in info])
    assert logit_new.grad is None
    assert value_new.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert isinstance(value_new.grad, torch.Tensor)
