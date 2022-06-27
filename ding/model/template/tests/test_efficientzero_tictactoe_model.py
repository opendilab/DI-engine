import pytest
from itertools import product
import torch
from ding.model.template.efficientzero import EfficientZeroNet
from ding.model.template.efficientzero.efficientzero_model import RepresentationNetwork, DynamicsNetwork
from ding.torch_utils import is_differentiable

bs_args = [10]

num_blocks = [3]
num_channels = [3]
reduced_channels_reward = [2]
fc_reward_layers = [[16, 8]]
full_support_size = [2]
block_output_size_reward = [180]
# dyn_args = [num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size, block_output_size_reward]
dyn_args = list(
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

    @pytest.mark.parametrize('batch_size', [(10)])
    def test_RepresentationNetwork(self, batch_size):
        batch = batch_size
        obs = torch.rand(batch, 1, 3, 3)
        repnet = RepresentationNetwork()
        state = repnet(obs)
        assert state.size() == obs.size()

    @pytest.mark.parametrize(
        'num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size,block_output_size_reward',
        dyn_args
    )
    def test_DynamicsNetwork(
        self, num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size,
        block_output_size_reward
    ):
        batch = 100  # this is (torch.randn(1, 10, 64), torch.randn(1, 10, 64)) => 100 / 10 = 10
        state = torch.rand(batch, 3, 3, 3)
        dynnet = DynamicsNetwork(
            num_blocks=num_blocks,
            num_channels=num_channels,
            reduced_channels_reward=reduced_channels_reward,
            fc_reward_layers=fc_reward_layers,
            full_support_size=full_support_size,
            block_output_size_reward=block_output_size_reward
        )
        state_, reward_hidden, value_prefix = dynnet(state, (torch.randn(1, 10, 64), torch.randn(1, 10, 64)))
        assert state_.size() == torch.Size([100, 2, 3, 3])
        # assert state_.size() == state.size()
