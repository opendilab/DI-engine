import torch
import numpy as np
import pytest
from itertools import product

from nervex.model.template import QAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

B = 4
T = 6
embedding_size = 32
action_shape_args = [(6, ), [
    1,
]]
args = list(product(*[action_shape_args, [True, False]]))


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape, twin', args)
class TestQAC:

    @staticmethod
    def output_check(action_shape, models, outputs):
        if isinstance(action_shape, tuple) or isinstance(action_shape, list):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, models)

    def test_fcqac(self, action_shape, twin):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = QAC(
            obs_shape=(N, ),
            action_shape=action_shape,
            critic_head_hidden_size=embedding_size,
            actor_head_hidden_size=embedding_size,
            twin_critic=twin,
        )
        # compute_q
        q = model(inputs, mode='compute_critic')['q_value']
        if twin:
            self.output_check(action_shape, model.critic[0], q[0])
            self.output_check(action_shape, model.critic[1], q[1])
        else:
            self.output_check(action_shape, model.critic, q)

        # compute_action
        action = model(inputs['obs'], mode='compute_actor')['action']
        if squeeze(action_shape) == 1:
            assert action.shape == (B, )
        else:
            assert action.shape == (B, squeeze(action_shape))
        assert action.eq(action.clamp(-1, 1)).all()
