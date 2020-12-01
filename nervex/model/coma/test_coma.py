import pytest
import torch
from nervex.torch_utils import is_differentiable
from .coma import ComaCriticNetwork


@pytest.mark.unittest
def test_coma():
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 32
    coma_model = ComaCriticNetwork(
        obs_dim + global_obs_dim + action_dim * 2, action_dim
    )
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, global_obs_dim),
        },
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num)),
        'last_action': torch.randint(0, action_dim, size=(T, bs, agent_num))
    }
    output = coma_model(data)
    assert set(output.keys()) == set(['agent_q'])
    assert output['agent_q'].shape == (T, bs, agent_num, action_dim)
    loss = output['agent_q'].sum() 
    is_differentiable(loss, coma_model)

