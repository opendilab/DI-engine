import pytest
import numpy as np
import torch
from itertools import product

from ding.model import mavac
from ding.model.template.mavac import MAVAC
from ding.torch_utils import is_differentiable

B = 32
agent_obs_shape = [216, 265]
global_obs_shape = [264, 324]
agent_num = 8
action_shape = 14
args = list(product(*[agent_obs_shape, global_obs_shape]))


@pytest.mark.unittest
@pytest.mark.parametrize('agent_obs_shape, global_obs_shape', args)
class TestVAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_vac(self, agent_obs_shape, global_obs_shape):
        data = {
            'agent_state': torch.randn(B, agent_num, agent_obs_shape),
            'global_state': torch.randn(B, agent_num, global_obs_shape),
            'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
        }
        model = MAVAC(agent_obs_shape, global_obs_shape, action_shape, agent_num)

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
        assert value.shape == (B, agent_num)
        self.output_check(model.critic, value, action_shape)
