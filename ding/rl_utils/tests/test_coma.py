import pytest
from itertools import product
import numpy as np
import torch
from ding.rl_utils import coma_data, coma_error

random_weight = torch.rand(128, 4, 8) + 1
weight_args = [None, random_weight]


@pytest.mark.unittest
@pytest.mark.parametrize('weight, ', weight_args)
def test_coma(weight):
    T, B, A, N = 128, 4, 8, 32
    logit = torch.randn(
        T,
        B,
        A,
        N,
    ).requires_grad_(True)
    action = torch.randint(
        0, N, size=(
            T,
            B,
            A,
        )
    )
    reward = torch.rand(T, B)
    q_value = torch.randn(T, B, A, N).requires_grad_(True)
    target_q_value = torch.randn(T, B, A, N).requires_grad_(True)
    mask = torch.randint(0, 2, (T, B, A))
    data = coma_data(logit, action, q_value, target_q_value, reward, weight)
    loss = coma_error(data, 0.99, 0.95)
    assert all([l.shape == tuple() for l in loss])
    assert logit.grad is None
    assert q_value.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit.grad, torch.Tensor)
    assert isinstance(q_value.grad, torch.Tensor)
