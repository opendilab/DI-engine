import torch
import numpy as np
import pytest

from nervex.model import FCQAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

B = 4
T = 6
embedding_dim = 32
action_dim_args = [(6, ), [1, ]]


@pytest.mark.unittest
@pytest.mark.parametrize('action_dim', action_dim_args)
class TestQAC:

    @staticmethod
    def output_check(action_dim, models, outputs):
        if isinstance(action_dim, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_dim):
            loss = outputs.sum()
        is_differentiable(loss, models)

    def test_fcqac(self, action_dim):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'act': torch.randn(B, squeeze(action_dim))}
        for twin in [False, True]:
            model = FCQAC(
                obs_dim=(N,),
                action_dim=action_dim,
                action_range={'min': -2, 'max': 2},
                state_action_embedding_dim=embedding_dim,
                state_embedding_dim=embedding_dim,
                head_hidden_dim=128,
                use_twin_critic=twin
            )
            # compute_q
            q = model(inputs, mode='compute_q')['q']
            self.output_check(model._act_dim, [model._critic_encoder[0], model._critic[0]], q[0])
            # compute_action
            action = model(inputs, mode='compute_action')['action']
            assert action.shape == (B, model._act_dim)
            assert action.eq(action.clamp(-2, 2)).all()
            # optimize_actor
            actor_loss_pos = model(inputs, mode='optimize_actor')['q']
            assert isinstance(actor_loss_pos, torch.Tensor)
            # self.output_check(model._act_dim, [model._actor_encoder, model._actor], -actor_loss_pos[0])
            # after optimize_actor
            q = model(inputs, mode='compute_q')['q']
            # self.output_check(model._act_dim, [model._critic_encoder[0], model._critic[0]], q[0])
