import pytest
from itertools import product
import torch
from ding.model.template.efficientzero.efficientzero_model import DynamicsNetwork
from ding.model.template.efficientzero.efficientzero_model import RepresentationNetwork
from ding.torch_utils import is_differentiable

bs_args = [10]

num_blocks = [3]
num_channels = [3]
reduced_channels_reward = [2]
fc_reward_layers = [[16, 8]]
full_support_size = [2]
block_output_size_reward = [180]
dynamics_network_args = list(
    product(
        num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size, block_output_size_reward
    )
)


@pytest.mark.unittest
class TestEfficientZero:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('batch_size', [10])
    def test_representation_network(self, batch_size):
        batch = batch_size
        obs = torch.rand(batch, 1, 3, 3)
        representation_network = RepresentationNetwork(observation_shape=[1, 3, 3], num_blocks=1, num_channels=16,
                                                       downsample=False)
        state = representation_network(obs)
        assert state.shape == torch.Size([10, 16, 3, 3])

    @pytest.mark.parametrize(
        'num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size,'
        'block_output_size_reward',
        dynamics_network_args
    )
    def test_dynamics_network(
            self, num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size,
            block_output_size_reward
    ):
        batch = 100
        state = torch.rand(batch, 3, 3, 3)
        dynamics_network = DynamicsNetwork(
            num_blocks=num_blocks,
            num_channels=num_channels,
            reduced_channels_reward=reduced_channels_reward,
            fc_reward_layers=fc_reward_layers,
            full_support_size=full_support_size,
            block_output_size_reward=block_output_size_reward
        )
        state, reward_hidden, value_prefix = dynamics_network(state, (torch.randn(1, 10, 64), torch.randn(1, 10, 64)))
        assert state.shape == torch.Size([100, 2, 3, 3])
        assert reward_hidden[0].shape == torch.Size([1, 10, 64])
        assert reward_hidden[1].shape == torch.Size([1, 10, 64])
        assert value_prefix.shape == torch.Size([10, 2])
