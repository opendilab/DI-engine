import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template import MADQN


@pytest.mark.unittest
def test_madqn():
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    madqn_model = MADQN(
        agent_num=agent_num,
        obs_shape=obs_dim,
        action_shape=action_dim,
        hidden_size_list=[embedding_dim, embedding_dim],
        global_obs_shape=global_obs_dim
    )
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, agent_num, global_obs_dim),
            'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
        },
        'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
    }
    output = madqn_model(data, cooperation=True, single_step=False)
    assert output['total_q'].shape == (T, bs)
    assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
