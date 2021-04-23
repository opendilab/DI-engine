import pytest
from itertools import product
import numpy as np
import torch
from nervex.rl_utils import ppg_data, ppg_joint_error

use_value_clip_args = [True, False]
random_weight = torch.rand(4) + 1
weight_args = [None, random_weight]
args = [item for item in product(*[use_value_clip_args, weight_args])]


@pytest.mark.unittest
@pytest.mark.parametrize('use_value_clip, weight', args)
def test_ppg(use_value_clip, weight):
    B, N = 4, 32
    logit_new = torch.randn(B, N).add_(0.1).clamp_(0.1, 0.99)
    logit_old = logit_new.add_(torch.rand_like(logit_new) * 0.1).clamp_(0.1, 0.99)
    logit_new.requires_grad_(True)
    logit_old.requires_grad_(True)
    action = torch.randint(0, N, size=(B, ))
    value_new = torch.randn(B).requires_grad_(True)
    value_old = value_new + torch.rand_like(value_new) * 0.1
    return_ = torch.randn(B) * 2
    data = ppg_data(logit_new, logit_old, action, value_new, value_old, return_, weight)
    loss = ppg_joint_error(data, use_value_clip=use_value_clip)
    assert all([l.shape == tuple() for l in loss])
    assert logit_new.grad is None
    assert value_new.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)
    assert isinstance(value_new.grad, torch.Tensor)
