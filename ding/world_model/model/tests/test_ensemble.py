import pytest
import torch
from itertools import product
from ding.world_model.model.ensemble import EnsembleFC, EnsembleModel

# arguments
state_size = [16]
action_size = [16, 1]
reward_size = [1]
args = list(product(*[state_size, action_size, reward_size]))


@pytest.mark.unittest
def test_EnsembleFC():
    in_dim, out_dim, ensemble_size, B = 4, 8, 7, 64
    fc = EnsembleFC(in_dim, out_dim, ensemble_size)
    x = torch.randn(ensemble_size, B, in_dim)
    y = fc(x)
    assert y.shape == (ensemble_size, B, out_dim)


@pytest.mark.parametrize('state_size, action_size, reward_size', args)
def test_EnsembleModel(state_size, action_size, reward_size):
    ensemble_size, B = 7, 64
    model = EnsembleModel(state_size, action_size, reward_size, ensemble_size)
    x = torch.randn(ensemble_size, B, state_size + action_size)
    y = model(x)
    assert len(y) == 2
    assert y[0].shape == y[1].shape == (ensemble_size, B, state_size + reward_size)
