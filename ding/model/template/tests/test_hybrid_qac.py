import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import QAC
from ding.torch_utils import is_differentiable
from ding.utils import squeeze
from easydict import EasyDict
B = 4
T = 6
embedding_size = 32
hybrid_args = {
        'action_shape': EasyDict({'action_type_shape':(4,), 'action_args_shape': (6,)}),
        'twin': True,
        'actor_head_type': 'hybrid'
    }


@pytest.mark.unittest
class TestHybridQAC:

    def test_hybrid_qac(self, action_shape = hybrid_args['action_shape'], 
                              twin = hybrid_args['twin'], 
                              actor_head_type = hybrid_args['actor_head_type']):
        N = 32
        assert actor_head_type == 'hybrid'
        inputs = {'obs': torch.randn(B, N), 
                      'action': [torch.rand(B, N), torch.rand(B, N, squeeze(action_shape.action_args_shape))], 
                      'logit':torch.randn(B, N, squeeze(action_shape.action_type_shape))
                      }
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

        discrete_logit = model(inputs['obs'], mode='compute_actor')['logit']
        continuous_args = model(inputs['obs'], mode='compute_actor')['action_args']
        # test discrete action
        if squeeze(action_shape.action_type_shape) == 1:
            assert discrete_logit.shape == (B, )
        else:
            assert discrete_logit.shape == (B, squeeze(action_shape.action_type_shape))
        is_differentiable(discrete_logit.sum(), model.actor)

        # test continuous action
        if squeeze(action_shape.action_args_shape) == 1:
            assert continuous_args.shape == (B, )
        else:
            assert continuous_args.shape == (B, squeeze(action_shape.action_args_shape))
        
        assert continuous_args.eq(continuous_args.clamp(-1, 1)).all()
        is_differentiable(continuous_args.sum(), model.actor)