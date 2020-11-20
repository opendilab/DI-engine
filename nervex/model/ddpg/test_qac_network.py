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
        model = FCQAC(
            obs_dim=(N, ),
            action_dim=action_dim,
            action_range={'min': -2, 'max': 2},
            state_action_embedding_dim=embedding_dim,
            state_embedding_dim=embedding_dim,
            head_hidden_dim=128,
            use_twin_critic=False
        )

        q = model(inputs, mode='compute_q')['q']
        self.output_check(model._act_dim, [model._critic_encoder[0], model._critic[0]], q[0])
        action = model(inputs, mode='compute_action')['action']
        assert action.shape == (B, model._act_dim)
        assert action.eq(action.clamp(-2, 2)).all()
        actor_loss_pos = model(inputs, mode='optimize_actor')['q']
        self.output_check(model._act_dim, [model._actor_encoder, model._actor], -actor_loss_pos[0])

    # def test_convdqn(self, action_dim):
    #     dims = [3, 64, 64]
    #     inputs = torch.randn(B, *dims)
    #     model = ConvDQN(dims, action_dim, embedding_dim)
    #     outputs = model(inputs)['logit']
    #     self.output_check(model, outputs)
    #
    # def test_fcdrqn(self, action_dim):
    #     N = 32
    #     data = torch.randn(T, B, N)
    #     model = FCDRQN((N, ), action_dim, embedding_dim)
    #     prev_state = [None for _ in range(B)]
    #     for t in range(T):
    #         inputs = {'obs': data[t], 'prev_state': prev_state}
    #         outputs = model(inputs)
    #         logit, prev_state = outputs['logit'], outputs['next_state']
    #         assert len(prev_state) == B
    #         assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
    #     # test the last step can backward correctly
    #     self.output_check(model, logit)
    #
    #     model = FCDRQN((N, ), action_dim, embedding_dim)
    #     data = torch.randn(T, B, N)
    #     prev_state = [None for _ in range(B)]
    #     inputs = {'obs': data, 'prev_state': prev_state, 'enable_fast_timestep': True}
    #     outputs = model(inputs)
    #     logit, prev_state = outputs['logit'], outputs['next_state']
    #     assert len(prev_state) == B
    #     assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
    #     self.output_check(model, logit)
    #     action_dim = model._head.action_dim
    #     if isinstance(action_dim, tuple):
    #         assert all([l.shape == (T, B, d) for l, d in zip(logit, action_dim)])
    #     elif np.isscalar(action_dim):
    #         assert logit.shape == (T, B, action_dim)
    #
    # def test_convdrqn(self, action_dim):
    #     dims = [3, 64, 64]
    #     data = torch.randn(T, B, *dims)
    #     model = ConvDRQN(dims, action_dim, embedding_dim)
    #     prev_state = [None for _ in range(B)]
    #     for t in range(T):
    #         inputs = {'obs': data[t], 'prev_state': prev_state}
    #         outputs = model(inputs)
    #         logit, prev_state = outputs['logit'], outputs['next_state']
    #         assert len(prev_state) == B
    #         assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
    #     # test the last step can backward correctly
    #     self.output_check(model, logit)
    #
    #     data = torch.randn(T, B, *dims)
    #     model = ConvDRQN(dims, action_dim, embedding_dim)
    #     prev_state = [None for _ in range(B)]
    #     inputs = {'obs': data, 'prev_state': prev_state, 'enable_fast_timestep': True}
    #     outputs = model(inputs)
    #     logit, prev_state = outputs['logit'], outputs['next_state']
    #     assert len(prev_state) == B
    #     assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
    #     self.output_check(model, logit)
    #     action_dim = model._head.action_dim
    #     if isinstance(action_dim, tuple):
    #         assert all([l.shape == (T, B, d) for l, d in zip(logit, action_dim)])
    #     elif np.isscalar(action_dim):
    #         assert logit.shape == (T, B, action_dim)
