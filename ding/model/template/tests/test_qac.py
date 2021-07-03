import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import QAC
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
@pytest.mark.parametrize('action_shape, twin, actor_head_type', args)
class TestQAC:

    def test_fcqac(self, action_shape, twin, actor_head_type):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = QAC(
            obs_shape=(N, ),
            action_shape=action_shape,
            actor_head_type=actor_head_type,
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
        if actor_head_type == 'regression':
            action = model(inputs['obs'], mode='compute_actor')['action']
            if squeeze(action_shape) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, squeeze(action_shape))
            assert action.eq(action.clamp(-1, 1)).all()
            is_differentiable(action.sum(), model.actor)
        elif actor_head_type == 'reparameterization':
            (mu, sigma) = model(inputs['obs'], mode='compute_actor')['logit']
            assert mu.shape == (B, *action_shape)
            assert sigma.shape == (B, *action_shape)
            is_differentiable(mu.sum() + sigma.sum(), model.actor)
