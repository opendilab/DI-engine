import pytest
import numpy as np
import torch
from itertools import product

from ding.model import VAC
from ding.torch_utils import is_differentiable

B, C, H, W = 4, 3, 128, 128
obs_shape = [4, (8, ), (4, 64, 64)]
act_args = [[6, False], [(3, ), True], [[2, 3, 6], False]]
#act_args = [[(3, ), True]]
args = list(product(*[obs_shape, act_args, [False, True]]))


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, act_args, share_encoder', args)
class TestVAC:

    def output_check(self, model, outputs, action_shape):
        if isinstance(action_shape, tuple):
            loss = sum([t.sum() for t in outputs])
        elif np.isscalar(action_shape):
            loss = outputs.sum()
        is_differentiable(loss, model)

    def test_vac(self, obs_shape, act_args, share_encoder):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = VAC(obs_shape, action_shape=act_args[0], continuous=act_args[1], share_encoder=share_encoder)

        outputs = model(inputs, mode='compute_actor_critic')
        value, logit = outputs['value'], outputs['logit']
        if model.continuous:
            outputs = value.sum() + logit[0].sum() + logit[1].sum()
        else:
            if model.multi_head:
                outputs = value.sum() + sum([t.sum() for t in logit])
            else:
                outputs = value.sum() + logit.sum()
        self.output_check(model, outputs, 1)

        for p in model.parameters():
            p.grad = None
        logit = model(inputs, mode='compute_actor')['logit']
        if model.continuous:
            logit = logit[0].sum() + logit[1].sum()
        self.output_check(model.actor, logit, model.action_shape)

        for p in model.parameters():
            p.grad = None
        value = model(inputs, mode='compute_critic')['value']
        assert value.shape == (B, )
        self.output_check(model.critic, value, 1)
