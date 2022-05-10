import pytest
import torch
from ding.rl_utils import get_epsilon_greedy_fn, create_noise_generator


@pytest.mark.unittest
def test_eps_greedy():
    exp_eps = get_epsilon_greedy_fn(start=0.9, end=0.1, decay=100)
    assert exp_eps(0) == 0.9
    assert exp_eps(10) > exp_eps(200)
    lin_eps1 = get_epsilon_greedy_fn(start=1.0, end=0.1, decay=90, type_='linear')
    assert lin_eps1(9) == 0.91
    assert lin_eps1(100) == 0.1
    lin_eps2 = get_epsilon_greedy_fn(start=0.9, end=0.3, decay=20, type_='linear')
    assert pytest.approx(lin_eps2(9)) == 0.63
    assert lin_eps2(100) == 0.3


@pytest.mark.unittest
def test_noise():
    bs, dim = 4, 15
    logits = torch.Tensor(bs, dim)
    gauss = create_noise_generator(noise_type='gauss', noise_kwargs={'mu': 0.0, 'sigma': 1.5})
    g_noise = gauss(logits.shape, logits.device)
    assert g_noise.shape == logits.shape
    assert g_noise.device == logits.device

    x0 = torch.rand(bs, dim)
    ou = create_noise_generator(noise_type='ou', noise_kwargs={'mu': 0.1, 'sigma': 1.0, 'theta': 2.0, 'x0': x0})
    o_noise1 = ou((bs, dim), x0.device)
    o_noise2 = ou((bs, dim), x0.device)
    assert o_noise2.shape == x0.shape
    assert o_noise2.device == x0.device
    assert not torch.equal(ou.x0, ou._x)  # OUNoise._x is not the same as _x0 after 2 calls
    assert torch.abs(x0 - ou.x0).max() < 1e-6  # OUNoise._x0 does not change
    x0 += 0.05
    ou.x0 = x0
    assert torch.abs(ou.x0 - x0).max() < 1e-6 and torch.abs(ou.x0 - ou._x).max() < 1e-6
    o_noise3 = ou(x0.shape, x0.device)
