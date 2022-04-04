import pytest
import torch
from ding.rl_utils import compute_q_retraces


@pytest.mark.unittest
def test_compute_q_retraces():
    T, B, N = 64, 32, 6
    q_values = torch.randn(T + 1, B, N)
    v_pred = torch.randn(T + 1, B, 1)
    rewards = torch.randn(T, B)
    ratio = torch.rand(T, B, N) * 0.4 + 0.8
    assert ratio.max() <= 1.2 and ratio.min() >= 0.8
    weights = torch.rand(T, B)
    actions = torch.randint(0, N, size=(T, B))
    with torch.no_grad():
        q_retraces = compute_q_retraces(q_values, v_pred, rewards, actions, weights, ratio, gamma=0.99)
    assert q_retraces.shape == (T + 1, B, 1)
