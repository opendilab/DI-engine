import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import MAQAC, ContinuousMAQAC
from ding.torch_utils import is_differentiable
from ding.utils.default_helper import squeeze

B = 32
agent_obs_shape = [216, 265]
global_obs_shape = [264, 324]
agent_num = 8
action_shape = 14
args = list(product(*[agent_obs_shape, global_obs_shape, [False, True]]))


@pytest.mark.unittest
@pytest.mark.parametrize('agent_obs_shape, global_obs_shape, twin_critic', args)
class TestMAQAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_maqac(self, agent_obs_shape, global_obs_shape, twin_critic):
        data = {
            'obs': {
                'agent_state': torch.randn(B, agent_num, agent_obs_shape),
                'global_state': torch.randn(B, agent_num, global_obs_shape),
                'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            }
        }
        model = MAQAC(agent_obs_shape, global_obs_shape, action_shape, twin_critic=twin_critic)

        logit = model(data, mode='compute_actor')['logit']
        value = model(data, mode='compute_critic')['q_value']

        value_sum = sum(t.sum() for t in value) if twin_critic else value.sum()
        outputs = value_sum + logit.sum()
        self.output_check(model, outputs, action_shape)

        for p in model.parameters():
            p.grad = None
        logit = model(data, mode='compute_actor')['logit']
        self.output_check(model.actor, logit, action_shape)

        for p in model.parameters():
            p.grad = None
        value = model(data, mode='compute_critic')['q_value']
        if twin_critic:
            for v in value:
                assert v.shape == (B, agent_num, action_shape)
        else:
            assert value.shape == (B, agent_num, action_shape)
        self.output_check(model.critic, sum(t.sum() for t in value) if twin_critic else value.sum(), action_shape)


B = 32
agent_obs_shape = [216, 265]
global_obs_shape = [264, 324]
agent_num = 8
action_shape = 14
action_space = ['regression', 'reparameterization']
args = list(product(*[agent_obs_shape, global_obs_shape, action_space, [False, True]]))


@pytest.mark.unittest
@pytest.mark.parametrize('agent_obs_shape, global_obs_shape, action_space, twin_critic', args)
class TestContinuousMAQAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_continuousmaqac(self, agent_obs_shape, global_obs_shape, action_space, twin_critic):
        data = {
            'obs': {
                'agent_state': torch.randn(B, agent_num, agent_obs_shape),
                'global_state': torch.randn(B, agent_num, global_obs_shape),
                'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            },
            'action': torch.randn(B, agent_num, squeeze(action_shape))
        }
        model = ContinuousMAQAC(agent_obs_shape, global_obs_shape, action_shape, action_space, twin_critic=twin_critic)

        for p in model.parameters():
            p.grad = None

        if action_space == 'regression':
            action = model(data['obs'], mode='compute_actor')['action']
            if squeeze(action_shape) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, agent_num, squeeze(action_shape))
            assert action.eq(action.clamp(-1, 1)).all()
            self.output_check(model.actor, action, action_shape)
            #is_differentiable(action.sum(), model.actor)
        elif action_space == 'reparameterization':
            (mu, sigma) = model(data['obs'], mode='compute_actor')['logit']
            assert mu.shape == (B, agent_num, action_shape)
            assert sigma.shape == (B, agent_num, action_shape)
            is_differentiable(mu.sum() + sigma.sum(), model.actor)

        for p in model.parameters():
            p.grad = None
        value = model(data, mode='compute_critic')['q_value']
        if twin_critic:
            for v in value:
                assert v.shape == (B, agent_num)
        else:
            assert value.shape == (B, agent_num)
        self.output_check(model.critic, sum(t.sum() for t in value) if twin_critic else value.sum(), action_shape)
