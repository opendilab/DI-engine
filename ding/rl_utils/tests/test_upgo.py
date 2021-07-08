import pytest
import torch
from ding.rl_utils.upgo import upgo_loss, upgo_returns, tb_cross_entropy


@pytest.mark.unittest
def test_upgo():
    T, B, N, N2 = 4, 8, 5, 7

    # tb_cross_entropy: 3 tests
    logit = torch.randn(T, B, N, N2).softmax(-1).requires_grad_(True)
    action = logit.argmax(-1).detach()
    ce = tb_cross_entropy(logit, action)
    assert ce.shape == (T, B)

    logit = torch.randn(T, B, N, N2, 2).softmax(-1).requires_grad_(True)
    action = logit.argmax(-1).detach()
    with pytest.raises(AssertionError):
        ce = tb_cross_entropy(logit, action)

    logit = torch.randn(T, B, N).softmax(-1).requires_grad_(True)
    action = logit.argmax(-1).detach()
    ce = tb_cross_entropy(logit, action)
    assert ce.shape == (T, B)

    # upgo_returns
    rewards = torch.randn(T, B)
    bootstrap_values = torch.randn(T + 1, B).requires_grad_(True)
    returns = upgo_returns(rewards, bootstrap_values)
    assert returns.shape == (T, B)

    # upgo loss
    rhos = torch.randn(T, B)
    loss = upgo_loss(logit, rhos, action, rewards, bootstrap_values)
    assert logit.requires_grad
    assert bootstrap_values.requires_grad
    for t in [logit, bootstrap_values]:
        assert t.grad is None
    loss.backward()
    for t in [logit]:
        assert isinstance(t.grad, torch.Tensor)
