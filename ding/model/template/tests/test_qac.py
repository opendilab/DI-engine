import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import ContinuousQAC, DiscreteMAQAC, DiscreteQAC
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
class TestContinuousQAC:

    def test_fcqac(self, action_shape, twin, action_space):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = ContinuousQAC(
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
            is_differentiable(q[0].sum(), model.critic[1][0])
            is_differentiable(q[1].sum(), model.critic[1][1])
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


args = list(product(*[[True, False], [(13, ), [4, 84, 84]]]))


@pytest.mark.unittest
@pytest.mark.parametrize('twin, obs_shape', args)
class TestDiscreteQAC:

    def test_discreteqac(self, twin, obs_shape):
        action_shape = 6
        inputs = torch.randn(B, *obs_shape)
        model = DiscreteQAC(
            obs_shape=obs_shape,
            action_shape=action_shape,
            twin_critic=twin,
            encoder_hidden_size_list=[32, 32, 64] if len(obs_shape) > 1 else None,
        )
        # compute_critic
        q = model(inputs, mode='compute_critic')['q_value']
        if twin:
            is_differentiable(q[0].sum(), model.critic[1][0])
            # is_differentiable(q[1].sum(), model.critic[1][1]) # backward encoder twice
            assert q[0].shape == (B, action_shape)
            assert q[1].shape == (B, action_shape)
        else:
            is_differentiable(q.sum(), model.critic[1])
            assert q.shape == (B, action_shape)

        # compute_actor
        print(model)
        logit = model(inputs, mode='compute_actor')['logit']
        assert logit.shape == (B, action_shape)
        is_differentiable(logit.sum(), model.actor)


B = 4
embedding_size = 64
action_shape_args = [(6, ), 1]
args = list(product(*[action_shape_args, [True, False], [True, False]]))


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape, twin, share_encoder', args)
class TestContinuousQACPixel:

    def test_qacpixel(self, action_shape, twin, share_encoder):
        inputs = {'obs': torch.randn(B, 3, 84, 84), 'action': torch.randn(B, squeeze(action_shape))}
        model = ContinuousQAC(
            obs_shape=(3, 84, 84),
            action_shape=action_shape,
            action_space='reparameterization',
            critic_head_hidden_size=embedding_size,
            actor_head_hidden_size=embedding_size,
            twin_critic=twin,
            share_encoder=share_encoder,
            encoder_hidden_size_list=[32, 32, 64],
        )
        # compute_q
        q = model(inputs, mode='compute_critic')['q_value']
        if twin:
            q = torch.min(q[0], q[1])
        is_differentiable(q.sum(), model.critic)

        # compute_action
        print(model)
        (mu, sigma) = model(inputs['obs'], mode='compute_actor')['logit']
        action_shape = squeeze(action_shape)
        assert mu.shape == (B, action_shape)
        assert sigma.shape == (B, action_shape)
        if share_encoder:  # if share_encoder, actor_encoder's grad is not None
            is_differentiable(mu.sum() + sigma.sum(), model.actor_head)
        else:
            is_differentiable(mu.sum() + sigma.sum(), model.actor)
