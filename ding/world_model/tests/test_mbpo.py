import pytest
import torch

from itertools import product
from easydict import EasyDict
from ding.world_model.mbpo import MBPOWorldModel
from ding.utils import deep_merge_dicts

# arguments
state_size = [16]
action_size = [16, 1]
args = list(product(*[state_size, action_size]))


@pytest.mark.unittest
class TestMBPO:

    def get_world_model(self, state_size, action_size):
        cfg = MBPOWorldModel.default_config()
        cfg.model.max_epochs_since_update = 0
        cfg = deep_merge_dicts(
            cfg, dict(cuda=False, model=dict(state_size=state_size, action_size=action_size, reward_size=1))
        )
        fake_env = EasyDict(termination_fn=lambda obs: torch.zeros_like(obs.sum(-1)).bool())
        return MBPOWorldModel(cfg, fake_env, None)

    @pytest.mark.parametrize('state_size, action_size', args)
    def test_step(self, state_size, action_size):
        states = torch.rand(128, state_size)
        actions = torch.rand(128, action_size)
        model = self.get_world_model(state_size, action_size)
        model.elite_model_idxes = [0, 1]
        rewards, next_obs, dones = model.step(states, actions)
        assert rewards.shape == (128, )
        assert next_obs.shape == (128, state_size)
        assert dones.shape == (128, )

    @pytest.mark.parametrize('state_size, action_size', args)
    def test_train(self, state_size, action_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = self.get_world_model(state_size, action_size)
        model._train(inputs[:64], labels[:64])

    @pytest.mark.parametrize('state_size, action_size', args[:1])
    def test_others(self, state_size, action_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = self.get_world_model(state_size, action_size)
        model._train(inputs[:64], labels[:64])
        model._save_states()
        model._load_states()
        model._save_best(0, [1, 2, 3])
