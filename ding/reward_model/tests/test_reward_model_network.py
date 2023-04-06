from collections.abc import Iterable

import torch
import pytest
import torch.optim as optim
import torch.nn.functional as F

from ding.reward_model.network import RepresentationNetwork, RndNetwork, RedNetwork


@pytest.mark.unittest
def test_representation_network():
    # len(obs_shape) == 3
    obs_shape = [4, 84, 84]
    batch_size = 32
    hidden_size_list = [16, 16, 16, 16]
    reward_model = RepresentationNetwork(obs_shape, hidden_size_list)
    data = torch.randn([batch_size] + obs_shape)
    data_feature = reward_model(data)
    assert data_feature.shape == (batch_size, hidden_size_list[-1])

    # len(obs_shape) == 4
    with pytest.raises(KeyError):
        obs_shape = [4, 84, 84, 5]
        reward_model = RepresentationNetwork(obs_shape, hidden_size_list)
