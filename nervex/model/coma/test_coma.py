import pytest
import torch
from nervex.torch_utils import is_differentiable
from .coma import ComaCriticNetwork, ComaActorNetwork


@pytest.mark.unittest
def test_coma_critic():
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    coma_model = ComaCriticNetwork(obs_dim + global_obs_dim + action_dim, action_dim)
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, global_obs_dim),
        },
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num)),
    }
    output = coma_model(data)
    assert set(output.keys()) == set(['total_q'])
    assert output['total_q'].shape == (T, bs, agent_num, action_dim)
    loss = output['total_q'].sum()
    is_differentiable(loss, coma_model)


@pytest.mark.unittest
def test_rnn_actor_net():
    T, B, N = 4, 8, 32
    embedding_dim = 64
    action_dim = (6, )
    data = torch.randn(T, B, N)
    model = ComaActorNetwork((N, ), action_dim, embedding_dim)
    prev_state = [None for _ in range(B)]
    for t in range(T):
        inputs = {'obs': data[t], 'prev_state': prev_state}
        outputs = model(inputs)
        logit, prev_state = outputs['logit'], outputs['next_state']
        assert len(prev_state) == B
        assert all([len(o) == 2 and all([isinstance(o1, torch.Tensor) for o1 in o]) for o in prev_state])
    # test the last step can backward correctly
    loss = logit.sum()
    is_differentiable(loss, model)
