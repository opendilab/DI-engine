import torch
import numpy as np
import pytest

from nervex.model import QAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

B = 4
T = 6
embedding_dim = 32
action_dim_args = [(6, ), [
    1,
]]


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
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_dim))}
        for twin in [False, True]:
            model = QAC(
                obs_dim=(N, ),
                action_dim=action_dim,
                state_action_embedding_dim=embedding_dim,
                state_embedding_dim=embedding_dim,
                twin_critic=twin,
            )
            # compute_q
            q = model(inputs, mode='compute_critic')['q_value']
            if twin:
                self.output_check(model._act_dim, model._critic[0], q[0])
                self.output_check(model._act_dim, model._critic[1], q[1])
            else:
                self.output_check(model._act_dim, model._critic, q)

            # compute_action
            action = model(inputs['obs'], mode='compute_actor')['action']
            if squeeze(action_dim) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, squeeze(action_dim))
            assert action.eq(action.clamp(-1, 1)).all()

            # optimize_actor
            for p in model._critic.parameters():
                p.grad.zero_()
            inputs_oa = {'obs': torch.randn(B, N), 'action': action}
            actor_loss_pos = model(inputs_oa, mode='compute_critic')['q_value']
            if twin:
                actor_loss_pos = sum(actor_loss_pos)
            assert isinstance(actor_loss_pos, torch.Tensor)
            # actor has grad
            self.output_check(model._act_dim, model._actor, -actor_loss_pos)
            # critic does not have grad
            # for p in model._critic.parameters():
            #     assert p.grad.eq(0).all()

            # after optimize_actor
            if twin:
                for p in model._critic[0].parameters():
                    p.grad = None
            else:
                for p in model._critic.parameters():
                    p.grad = None
            q = model(inputs, mode='compute_critic')['q_value']
            if twin:
                self.output_check(model._act_dim, model._critic[0], q[0])
            else:
                self.output_check(model._act_dim, model._critic, q)
