import pytest
import torch

from itertools import product
from easydict import EasyDict
from ding.world_model.dreamer import DREAMERWorldModel
from ding.utils import deep_merge_dicts

# arguments
state_size = [[3, 64, 64]]
action_size = [6, 1]
args = list(product(*[state_size, action_size]))


@pytest.mark.unittest
class TestDREAMER:

    def get_world_model(self, state_size, action_size):
        cfg = DREAMERWorldModel.default_config()
        cfg.model.max_epochs_since_update = 0
        cfg = deep_merge_dicts(
            cfg, dict(cuda=False, model=dict(state_size=state_size, action_size=action_size, reward_size=1))
        )
        fake_env = EasyDict(termination_fn=lambda obs: torch.zeros_like(obs.sum(-1)).bool())
        return DREAMERWorldModel(cfg, fake_env, None)

    @pytest.mark.parametrize('state_size, action_size', args)
    def test_train(self, state_size, action_size):
        states = torch.rand(1280, *state_size)
        actions = torch.rand(1280, action_size)

        model = self.get_world_model(state_size, action_size)
