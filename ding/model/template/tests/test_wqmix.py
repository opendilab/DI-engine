import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template.wqmix import MixerStar, WQMix
#TEST

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
def test_wqmix():
    is_q_star = [True, False]
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    for q_star in is_q_star:
        wqmix_model = WQMix(agent_num, obs_dim, global_obs_dim, action_dim, [128, embedding_dim], 'gru', q_star)
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, agent_num, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
            },
            'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
            'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
        }
        output = wqmix_model(data, single_step=False)
        assert set(output.keys()) == set(['total_q', 'logit', 'next_state', 'action_mask'])
        assert output['total_q'].shape == (T, bs)
        assert output['logit'].shape == (T, bs, agent_num, action_dim)
        assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
        print(output['next_state'][0][0][0].shape)
        loss = output['total_q'].sum()
        is_differentiable(loss, wqmix_model)
        data.pop('action')
        output = wqmix_model(data, single_step=False)
