import torch
import numpy as np
import pytest

from nervex.model import QAC
from nervex.torch_utils import is_differentiable
from nervex.utils import squeeze

B = 4
T = 6
embedding_size = 32
action_shape_args = [(6, ), [
    1,
]]


@pytest.mark.unittest
@pytest.mark.parametrize('action_shape', action_shape_args)
class TestQAC:

    @staticmethod
    def output_check(action_shape, models, outputs):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, models)

    def test_fcqac(self, action_shape):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        for twin in [False, True]:
            model = QAC(
                obs_shape=(N, ),
                action_shape=action_shape,
                obs_action_embedding_size=embedding_size,
                obs_embedding_size=embedding_size,
                twin_critic=twin,
            )
            # compute_q
            q = model(inputs, mode='compute_critic')['q_value']
            if twin:
                self.output_check(model._act_shape, model._critic[0], q[0])
                self.output_check(model._act_shape, model._critic[1], q[1])
            else:
                self.output_check(model._act_shape, model._critic, q)

            # compute_action
            action = model(inputs['obs'], mode='compute_actor')['action']
            if squeeze(action_shape) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, squeeze(action_shape))
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
            self.output_check(model._act_shape, model._actor, -actor_loss_pos)
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
                self.output_check(model._act_shape, model._critic[0], q[0])
            else:
                self.output_check(model._act_shape, model._critic, q)
