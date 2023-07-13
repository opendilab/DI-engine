import pytest
import numpy as np
import torch
from itertools import product

from ding.model import mavac
from ding.model.template.mavac_diff_obs import MAVACDO
from ding.torch_utils import is_differentiable

B = 32
num_good = 1
num_adversaries = 3
num_obstacles = 2
agent_obs_shape = {
    'adversary_0':2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1),
    'adversary_1':2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1),
    'adversary_2':2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1),
    'agent_0': 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
}
global_obs_shape = [agent_obs_shape['adversary_0']+(2+2)*(num_good+num_adversaries)+2*num_obstacles,agent_obs_shape['agent_0']+(2+2)*(num_good+num_adversaries)+2*num_obstacles]
action_shape = 5
# args = list(product(*[agent_obs_shape, global_obs_shape]))

@pytest.mark.unittest
@pytest.mark.parametrize('agent_obs_shape, global_obs_shape', [(agent_obs_shape,global_obs_shape),])

class TestVAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_vac(self, agent_obs_shape, global_obs_shape):
        data = {
            'agent_state': {
                'adversary_0':torch.randn(B, agent_obs_shape['adversary_0']),
                'adversary_1':torch.randn(B, agent_obs_shape['adversary_1']),
                'adversary_2':torch.randn(B, agent_obs_shape['adversary_2']),
                'agent_0': torch.randn(B, agent_obs_shape['agent_0']),
            },
            # torch.randn(B, agent_num, agent_obs_shape),
            'global_state': {
                'adversary_0':torch.randn(B, global_obs_shape[0]),
                'adversary_1':torch.randn(B, global_obs_shape[0]),
                'adversary_2':torch.randn(B, global_obs_shape[0]),
                'agent_0': torch.randn(B, global_obs_shape[1]),
            },
            # torch.randn(B, agent_num, global_obs_shape),
            'action_mask': torch.randint(0, 2, size=(B, num_good+num_adversaries, action_shape))
        }
        model = MAVACDO(agent_obs_shape, global_obs_shape, action_shape, num_good, num_adversaries, num_obstacles)

        logit = model(data, mode='compute_actor_critic')['logit']
        value = model(data, mode='compute_actor_critic')['value']

        outputs = value.sum() + logit.sum()
        self.output_check(model, outputs, action_shape)

        for p in model.parameters():
            p.grad = None
        logit = model(data, mode='compute_actor')['logit']
        self.output_check(model.actor, logit, model.action_shape)

        for p in model.parameters():
            p.grad = None
        value = model(data, mode='compute_critic')['value']
        assert value.shape == (B, num_good+num_adversaries)
        self.output_check(model.critic, value, action_shape)

# test_vac(agent_obs_shape, global_obs_shape)