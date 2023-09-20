import pytest
from itertools import product
import torch
from ding.model.template import QTran
from ding.torch_utils import is_differentiable


@pytest.mark.unittest
def test_qtran():
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, global_obs_dim),
            'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
        },
        'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
    }
    model = QTran(agent_num, obs_dim, global_obs_dim, action_dim, [32, embedding_dim], embedding_dim)
    output = model.forward(data, single_step=False)
    assert set(output.keys()) == set(['next_state', 'agent_q_act', 'vs', 'logit', 'action_mask', 'total_q'])
    assert output['total_q'].shape == (T, bs)
    assert output['logit'].shape == (T, bs, agent_num, action_dim)
    assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
    print(output['next_state'][0][0]['h'].shape)
    loss = output['total_q'].sum() + output['agent_q_act'].sum() + output['vs'].sum()
    is_differentiable(loss, model)

    data.pop('action')
    outputs = model.forward(data, single_step=False)
