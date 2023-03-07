import torch
import numpy as np
import pytest
from itertools import product

from ding.model.template import PG
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4


@pytest.mark.unittest
class TestDiscretePG:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    def test_discrete_pg(self):
        obs_shape = (4, 84, 84)
        action_shape = 5
        model = PG(
            obs_shape,
            action_shape,
        )
        inputs = torch.randn(B, 4, 84, 84)

        outputs = model(inputs)
        assert isinstance(outputs, dict)
        assert outputs['logit'].shape == (B, action_shape)
        assert outputs['dist'].sample().shape == (B, )
        self.output_check(model, outputs['logit'])

    def test_continuous_pg(self):
        N = 32
        action_shape = (6, )
        inputs = {'obs': torch.randn(B, N), 'action': torch.randn(B, squeeze(action_shape))}
        model = PG(
            obs_shape=(N, ),
            action_shape=action_shape,
            action_space='continuous',
        )
        # compute_action
        print(model)
        outputs = model(inputs['obs'])
        assert isinstance(outputs, dict)
        dist = outputs['dist']
        action = dist.sample()
        assert action.shape == (B, *action_shape)

        logit = outputs['logit']
        mu, sigma = logit['mu'], logit['sigma']
        assert mu.shape == (B, *action_shape)
        assert sigma.shape == (B, *action_shape)
        is_differentiable(mu.sum() + sigma.sum(), model)
