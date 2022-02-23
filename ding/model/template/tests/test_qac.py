import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import QAC, MAQAC, DiscreteQAC
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4
T = 6
embedding_size = 32
action_shape_args = [(6, ), [
    1,
]]
args = list(product(*[action_shape_args, [True, False], ['regression', 'reparameterization']]))


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape, twin, action_space', args)
class TestQAC:

    def test_fcqac(self, action_shape, twin, action_space):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = QAC(
            obs_shape=(N, ),
            action_shape=action_shape,
            action_space=action_space,
            critic_head_hidden_size=embedding_size,
            actor_head_hidden_size=embedding_size,
            twin_critic=twin,
        )
        # compute_q
        q = model(inputs, mode='compute_critic')['q_value']
        if twin:
            is_differentiable(q[0].sum(), model.critic[0])
            is_differentiable(q[1].sum(), model.critic[1])
        else:
            is_differentiable(q.sum(), model.critic)

        # compute_action
        print(model)
        if action_space == 'regression':
            action = model(inputs['obs'], mode='compute_actor')['action']
            if squeeze(action_shape) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, squeeze(action_shape))
            assert action.eq(action.clamp(-1, 1)).all()
            is_differentiable(action.sum(), model.actor)
        elif action_space == 'reparameterization':
            (mu, sigma) = model(inputs['obs'], mode='compute_actor')['logit']
            assert mu.shape == (B, *action_shape)
            assert sigma.shape == (B, *action_shape)
            is_differentiable(mu.sum() + sigma.sum(), model.actor)


args = list(product(*[[True, False]]))


@pytest.mark.unittest
@pytest.mark.parametrize('twin', args)
class TestDiscreteQAC:

    def test_discreteqac(self, twin):
        N = 32
        A = 6
        inputs = {'obs': torch.randn(B, N)}
        model = DiscreteQAC(
            agent_obs_shape=N,
            global_obs_shape=N,
            action_shape=A,
            twin_critic=twin,
        )
        # compute_q
        q = model(inputs, mode='compute_critic')['q_value']
        if twin:
            is_differentiable(q[0].sum(), model.critic[0])
            is_differentiable(q[1].sum(), model.critic[1])
        else:
            is_differentiable(q.sum(), model.critic)

        # compute_action
        print(model)
        logit = model(inputs, mode='compute_actor')['logit']
        assert logit.shape[0] == B
        assert logit.shape[1] == A


B = 32
agent_obs_shape = [216, 265]
global_obs_shape = [264, 324]
agent_num = 8
action_shape = 14
args = list(product(*[agent_obs_shape, global_obs_shape]))


@pytest.mark.unittest
@pytest.mark.parametrize('agent_obs_shape, global_obs_shape', args)
class TestMAQAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_maqac(self, agent_obs_shape, global_obs_shape):
        data = {
            'obs': {
                'agent_state': torch.randn(B, agent_num, agent_obs_shape),
                'global_state': torch.randn(B, agent_num, global_obs_shape),
                'action_mask': torch.randint(0, 2, size=(B, agent_num, action_shape))
            }
        }
        model = MAQAC(agent_obs_shape, global_obs_shape, action_shape)

        logit = model(data, mode='compute_actor')['logit']
        value = model(data, mode='compute_critic')['q_value']

        outputs = value.sum() + logit.sum()
        self.output_check(model, outputs, action_shape)

        for p in model.parameters():
            p.grad = None
        logit = model(data, mode='compute_actor')['logit']
        self.output_check(model.actor, logit, action_shape)

        for p in model.parameters():
            p.grad = None
        value = model(data, mode='compute_critic')['q_value']
        assert value.shape == (B, agent_num, action_shape)
        self.output_check(model.critic, value, action_shape)
