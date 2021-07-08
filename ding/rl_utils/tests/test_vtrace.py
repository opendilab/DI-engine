import pytest
import torch
from ding.rl_utils import vtrace_data, vtrace_error


@pytest.mark.unittest
def test_vtrace():
    T, B, N = 4, 8, 16
    value = torch.randn(T + 1, B).requires_grad_(True)
    reward = torch.rand(T, B)
    target_output = torch.randn(T, B, N).requires_grad_(True)
    behaviour_output = torch.randn(T, B, N)
    action = torch.randint(0, N, size=(T, B))
    data = vtrace_data(target_output, behaviour_output, action, value, reward, None)
    loss = vtrace_error(data, rho_clip_ratio=1.1)
    assert all([l.shape == tuple() for l in loss])
    assert target_output.grad is None
    assert value.grad is None
    loss = sum(loss)
    loss.backward()
    assert isinstance(target_output, torch.Tensor)
    assert isinstance(value, torch.Tensor)
