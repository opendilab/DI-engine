import pytest
from itertools import product

import torch

from ding.model.template import ProcedureCloningMCTS, ProcedureCloningBFS

B = 4
T = 15
obs_shape = [(64, 64, 3)]
action_dim = [9]
obs_embeddings = 256
args = list(product(*[obs_shape, action_dim]))


@pytest.mark.unittest
@pytest.mark.parametrize('obs_shape, action_dim', args)
class TestProcedureCloning:

    def test_procedure_cloning_mcts(self, obs_shape, action_dim):
        inputs = {
            'states': torch.randn(B, *obs_shape),
            'goals': torch.randn(B, *obs_shape),
            'actions': torch.randn(B, T, action_dim)
        }
        model = ProcedureCloningMCTS(obs_shape=obs_shape, action_dim=action_dim)
        goal_preds, action_preds = model(inputs['states'], inputs['goals'], inputs['actions'])
        assert goal_preds.shape == (B, obs_embeddings)
        assert action_preds.shape == (B, T + 1, action_dim)

    def test_procedure_cloning_bfs(self, obs_shape, action_dim):
        o_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        model = ProcedureCloningBFS(obs_shape=o_shape, action_shape=action_dim)

        inputs = torch.randn(B, *obs_shape)
        map_preds = model(inputs)
        assert map_preds['logit'].shape == (B, obs_shape[0], obs_shape[1], action_dim + 1)
