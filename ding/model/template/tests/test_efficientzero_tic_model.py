import pytest
from itertools import product
import torch
from ding.model.template.model_based import EfficientZeroNet
from ding.model.template.model_based.efficientzero_tic_model_new import RepresentationNetwork, DynamicsNetwork
from ding.torch_utils import is_differentiable

bs_args = [10]

num_blocks = 2
num_channels = 2
reduced_channels_reward = 2
fc_reward_layers = 2
full_support_size = 1
block_output_size_reward = 16
dyn_args = [num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size, block_output_size_reward]


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
    def test_RepresentationNetwork(self, batch_size):
        batch = batch_size
        obs = torch.rand(batch, 3, 3, 3)
        repnet = RepresentationNetwork()
        state = repnet(obs)
        assert state.size() == obs.size()


    @pytest.mark.parametrize('num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size,block_output_size_reward', dyn_args)
    def test_DynamicsNetwork(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward
    ):
        batch = 10
        state = torch.rand(batch, 3, 3, 3)
        dynnet= DynamicsNetwork(dyn_args)
        state_, reward_hidden, value_prefix = dynnet(state, (5, 5))
        assert state_.size() == state.size()


