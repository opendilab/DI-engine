import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import QACDIST
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4
T = 6
embedding_size = 32
action_shape_args = [(6, ), [
    1,
]]
args = list(product(*[action_shape_args, ['regression', 'reparameterization']]))


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape, action_space', args)
class TestQACDIST:

    def test_fcqac_dist(self, action_shape, action_space):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = QACDIST(
            obs_shape=(N, ),
            action_shape=action_shape,
            action_space=action_space,
            critic_head_hidden_size=embedding_size,
            actor_head_hidden_size=embedding_size,
        )
        # compute_q
        q = model(inputs, mode='compute_critic')
        is_differentiable(q['q_value'].sum(), model.critic)

        if isinstance(action_shape, int):
            assert q['q_value'].shape == (B, 1)
            assert q['distribution'].shape == (B, 1, 51)
        elif len(action_shape) == 1:
            assert q['q_value'].shape == (B, 1)
            assert q['distribution'].shape == (B, 1, 51)

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
