import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template.wqmix import MixerStar, WQMix

args = [True, False]


@pytest.mark.unittest
def test_mixer_star():
    agent_num, bs, embedding_dim = 4, 3, 32
    agent_q = torch.randn(bs, agent_num)
    state_embedding = torch.randn(bs, embedding_dim)
    mixer_star = MixerStar(agent_num, embedding_dim, 64)
    total_q = mixer_star(agent_q, state_embedding)
    assert total_q.shape == (bs, )
    loss = total_q.mean()
    is_differentiable(loss, mixer_star)


@pytest.mark.unittest
@pytest.mark.parametrize('is_q_star', args)
def test_wqmix(is_q_star):
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    wqmix_model = WQMix(agent_num, obs_dim, global_obs_dim, action_dim, [128, embedding_dim], 'gru')
    data = {
        'obs': {
            'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            'global_state': torch.randn(T, bs, global_obs_dim),
            'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
        },
        'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
        'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
    }
    output = wqmix_model(data, single_step=False, q_star=is_q_star)
    assert set(output.keys()) == set(['total_q', 'logit', 'next_state', 'action_mask'])
    assert output['total_q'].shape == (T, bs)
    assert output['logit'].shape == (T, bs, agent_num, action_dim)
    assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
    print(output['next_state'][0][0]['h'].shape)
    loss = output['total_q'].sum()
    if is_q_star:
        is_differentiable(loss, [wqmix_model._q_network_star, wqmix_model._mixer_star])
    else:
        is_differentiable(loss, [wqmix_model._q_network, wqmix_model._mixer])
    data.pop('action')
    output = wqmix_model(data, single_step=False, q_star=is_q_star)
