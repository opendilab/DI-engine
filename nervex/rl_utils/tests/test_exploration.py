import pytest
import torch
from nervex.rl_utils import epsilon_greedy, GaussianNoise, OUNoise


@pytest.mark.unittest
def test_eps_greedy():
    exp_eps = epsilon_greedy(start=0.9, end=0.1, decay=100)
    assert exp_eps(0) == 0.9
    assert exp_eps(10) > exp_eps(200)
    lin_eps1 = epsilon_greedy(start=1.0, end=0.1, decay=90, type_='linear')
    assert lin_eps1(9) == 0.91
    assert lin_eps1(100) == 0.1
    with pytest.raises(Exception):
        lin_eps2 = epsilon_greedy(start=0.9, end=0.1, decay=20, type_='linear')


@pytest.mark.unittest
def test_noise():
    bs, dim = 4, 15
    logits = torch.Tensor(bs, dim)
    gauss = GaussianNoise(mu=0.0, sigma=1.5)
    g_noise = gauss(logits.shape, logits.device)
    assert g_noise.shape == logits.shape
    assert g_noise.device == logits.device

    x0 = torch.Tensor(bs, dim)
    ou = OUNoise(mu=0.1, sigma=1.0, theta=2.0, x0=x0)
    o_noise1 = ou((bs, dim), x0.device)
    o_noise2 = ou((bs, dim), x0.device)
    assert o_noise2.shape == x0.shape
    assert o_noise2.device == x0.device
    assert not torch.equal(ou.x0, ou._x)  # OUNoise._x is not the same as _x0 after 2 calls
    assert torch.equal(x0, ou.x0)  # OUNoise._x0 does not change
    x0 += 0.05
    ou.x0 = x0
    assert torch.equal(ou.x0, x0) and torch.equal(ou.x0, ou._x)
    o_noise3 = ou(x0.shape, x0.device)
