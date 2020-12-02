import pytest
from itertools import product
import numpy as np
import torch
from nervex.rl_utils import coma_data, coma_error

random_weight = torch.rand(128, 4, 8) + 1
weight_args = [None, random_weight]


@pytest.mark.unittest
@pytest.mark.parametrize('weight, ', weight_args)
def test_coma(weight):
    T, B, A, N = 128, 4, 8, 32
    logit = torch.randn(T, B, A, N,).requires_grad_(True)
    action = torch.randint(0, N, size=(T, B, A, ))
    q_val = torch.randn(T, B, A, N).requires_grad_(True)
    adv = torch.rand(T, B, A)
    return_ = torch.randn(T - 1, B, A) * 2
    mask = torch.randint(0, 2, (T, B, A))
    data = coma_data(logit, action, q_val, adv, return_, weight, mask)
    loss = coma_error(data)
    assert all([l.shape == tuple() for l in loss])
    assert logit.grad is None
    assert q_val.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit.grad, torch.Tensor)
    assert isinstance(q_val.grad, torch.Tensor)
