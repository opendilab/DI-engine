import pytest
from itertools import product
import torch
from ding.model.template.model_based import EnsembleDynamicsModel
from ding.torch_utils import is_differentiable

# constants
network_size = 3
elite_size = 2
hidden_size = 20
# arguments
state_size = [16]
action_size = [16]
reward_size = [1,2]
args = list(product(*[state_size, action_size, reward_size]))

@pytest.mark.unittest
class TestMBPO:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('state_size, action_size, reward_size', args)
    def test_mbpo(self, state_size, action_size, reward_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True).repeat(1, reward_size)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = EnsembleDynamicsModel(network_size, elite_size, state_size, action_size, reward_size, hidden_size, use_decay=True, cuda=False)

        model._train(inputs[:640], labels[:640])
