import pytest
import torch
from torch import nn

from itertools import product
from easydict import EasyDict
from ding.world_model.ddppo import DDPPOWorldMode, get_batch_jacobian, get_neighbor_index
from ding.utils import deep_merge_dicts

# arguments
state_size = [16]
action_size = [16, 1]
args = list(product(*[state_size, action_size]))


@pytest.mark.unittest
class TestDDPPO:

    def get_world_model(self, state_size, action_size):
        cfg = DDPPOWorldMode.default_config()
        cfg.model.max_epochs_since_update = 0
        cfg = deep_merge_dicts(
            cfg, dict(cuda=False, model=dict(state_size=state_size, action_size=action_size, reward_size=1))
        )
        fake_env = EasyDict(termination_fn=lambda obs: torch.zeros_like(obs.sum(-1)).bool())
        model = DDPPOWorldMode(cfg, fake_env, None)
        model.serial_calc_nn = True
        return model

    def test_get_neighbor_index(self):
        k = 2
        data = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, -1], [5, 0, 0], [5, 0, 1], [5, 0, -1]])
        idx = get_neighbor_index(data, k, serial=True)
        target_idx = torch.tensor([[2, 1], [0, 2], [0, 1], [5, 4], [3, 5], [3, 4]])
        assert (idx - target_idx).sum() == 0

    def test_get_batch_jacobian(self):
        B, in_dim, out_dim = 64, 4, 8
        net = nn.Linear(in_dim, out_dim)
        x = torch.randn(B, in_dim)
        jacobian = get_batch_jacobian(net, x, out_dim)
        assert jacobian.shape == (B, out_dim, in_dim)

    @pytest.mark.parametrize('state_size, action_size', args)
    def test_get_jacobian(self, state_size, action_size):
        B, ensemble_size = 64, 7
        model = self.get_world_model(state_size, action_size)
        train_input_reg = torch.randn(ensemble_size, B, state_size + action_size)
        jacobian = model._get_jacobian(model.gradient_model, train_input_reg)
        assert jacobian.shape == (ensemble_size, B, state_size + 1, state_size + action_size)
        assert jacobian.requires_grad

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
    def test_train_rollout_model(self, state_size, action_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True).repeat(1, 1)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = self.get_world_model(state_size, action_size)
        model._train_rollout_model(inputs[:64], labels[:64])

    @pytest.mark.parametrize('state_size, action_size', args)
    def test_train_graident_model(self, state_size, action_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = self.get_world_model(state_size, action_size)
        model._train_gradient_model(inputs[:64], labels[:64], inputs[:64], labels[:64])

    @pytest.mark.parametrize('state_size, action_size', args[:1])
    def test_others(self, state_size, action_size):
        states = torch.rand(1280, state_size)
        actions = torch.rand(1280, action_size)

        next_states = states + actions.mean(1, keepdim=True)
        rewards = next_states.mean(1, keepdim=True)

        inputs = torch.cat([states, actions], dim=1)
        labels = torch.cat([rewards, next_states], dim=1)

        model = self.get_world_model(state_size, action_size)
        model._train_rollout_model(inputs[:64], labels[:64])
        model._train_gradient_model(inputs[:64], labels[:64], inputs[:64], labels[:64])
        model._save_states()
        model._load_states()
        model._save_best(0, [1, 2, 3])
