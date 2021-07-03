import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template.coma import COMACriticNetwork, COMAActorNetwork


@pytest.mark.unittest
def test_coma_critic():
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    coma_model = COMACriticNetwork(obs_dim - action_dim + global_obs_dim + 2 * action_dim * agent_num, action_dim)
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, global_obs_dim),
        },
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num)),
    }
    output = coma_model(data)
    assert set(output.keys()) == set(['q_value'])
    assert output['q_value'].shape == (T, bs, agent_num, action_dim)
    loss = output['q_value'].sum()
    is_differentiable(loss, coma_model)


@pytest.mark.unittest
def test_rnn_actor_net():
    T, B, A, N = 4, 8, 3, 32
    embedding_dim = 64
    action_dim = 6
    data = torch.randn(T, B, A, N)
    model = COMAActorNetwork((N, ), action_dim, [128, embedding_dim])
    prev_state = [[None for _ in range(A)] for _ in range(B)]
    for t in range(T):
        inputs = {'obs': {'agent_state': data[t], 'action_mask': None}, 'prev_state': prev_state}
        outputs = model(inputs)
        logit, prev_state = outputs['logit'], outputs['next_state']
        assert len(prev_state) == B
        assert all([len(o) == A and all([len(o1) == 2 for o1 in o]) for o in prev_state])
        assert logit.shape == (B, A, action_dim)
    # test the last step can backward correctly
    loss = logit.sum()
    is_differentiable(loss, model)
