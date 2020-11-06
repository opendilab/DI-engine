import pytest
import torch
from nervex.rl_utils import gae_data, gae


@pytest.mark.unittest
def test_gae():
    T, B = 32, 4
    value = torch.randn(T + 1, B)
    reward = torch.randn(T, B)
    data = gae_data(value, reward)
    adv = gae(data)
    assert adv.shape == (T, B)
