import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import DiscreteBC, ContinuousBC
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
class TestContinuousBC:

    def test_continuous_bc(self, action_shape, action_space):
        N = 32
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = ContinuousBC(
            obs_shape=(N, ),
            action_shape=action_shape,
            action_space=action_space,
            actor_head_hidden_size=embedding_size,
        )
        # compute_action
        print(model)
        if action_space == 'regression':
            action = model(inputs['obs'])['action']
            if squeeze(action_shape) == 1:
                assert action.shape == (B, )
            else:
                assert action.shape == (B, squeeze(action_shape))
            assert action.eq(action.clamp(-1, 1)).all()
            is_differentiable(action.sum(), model.actor)
        elif action_space == 'reparameterization':
            (mu, sigma) = model(inputs['obs'])['logit']
            assert mu.shape == (B, *action_shape)
            assert sigma.shape == (B, *action_shape)
            is_differentiable(mu.sum() + sigma.sum(), model.actor)


T, B = 3, 4
obs_shape = [4, (8, ), (4, 64, 64)]
act_shape = [3, (6, ), [2, 3, 6]]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, act_shape', args)
class TestDiscreteBC:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    def test_discrete_bc(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = DiscreteBC(obs_shape, act_shape)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
        self.output_check(model, outputs['logit'])
