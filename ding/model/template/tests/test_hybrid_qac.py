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
    'action_shape': EasyDict({
        'action_type_shape': (4, ),
        'action_args_shape': (6, )
    }),
    'twin': True,
    'action_space': 'hybrid'
}


@pytest.mark.unittest
class TestHybridQAC:

    def test_hybrid_qac(
        self,
        action_shape=hybrid_args['action_shape'],
        twin=hybrid_args['twin'],
        action_space=hybrid_args['action_space']
    ):
        N = 32
        assert action_space == 'hybrid'
        inputs = {
            'obs': torch.randn(B, N),
            'action': {
                'action_type': torch.randint(0, squeeze(action_shape.action_type_shape), (B, )),
                'action_args': torch.rand(B, squeeze(action_shape.action_args_shape))
            },
            'logit': torch.randn(B, squeeze(action_shape.action_type_shape))
        }
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

        output = model(inputs['obs'], mode='compute_actor')
        discrete_logit = output['logit']
        continuous_args = output['action_args']
        # test discrete action_type + continuous action_args
        if squeeze(action_shape.action_type_shape) == 1:
            assert discrete_logit.shape == (B, )
        else:
            assert discrete_logit.shape == (B, squeeze(action_shape.action_type_shape))
        assert continuous_args.shape == (B, action_shape.action_args_shape)
        is_differentiable(discrete_logit.sum() + continuous_args.sum(), model.actor)
