import torch
import pytest
import numpy as np
from itertools import product

from ding.model.template import ProcedureCloning
from ding.torch_utils import is_differentiable
from ding.utils import squeeze

B = 4
T = 15
obs_shape = [(64, 64, 3)]
action_dim = [9]
obs_embeddings = 256
args = list(product(*[obs_shape, action_dim]))


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, action_dim', args)
class TestProcedureCloning:

    def test_procedure_cloning(self, obs_shape, action_dim):
        inputs = {'states': torch.randn(B, *obs_shape), 'goals': torch.randn(B, *obs_shape),\
             'actions': torch.randn(B, T, action_dim)}
        model = ProcedureCloning(obs_shape=obs_shape, action_dim=action_dim)

        print(model)

        goal_preds, action_preds = model(inputs['states'], inputs['goals'], inputs['actions'])
        assert goal_preds.shape == (B, obs_embeddings)
        assert action_preds.shape == (B, T + 1, action_dim)
