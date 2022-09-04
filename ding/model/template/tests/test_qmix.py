import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template import Mixer, QMix


@pytest.mark.unittest
def test_mixer():
    agent_num, bs, embedding_dim = 4, 3, 32
    agent_q = torch.randn(bs, agent_num)
    state_embedding = torch.randn(bs, embedding_dim)
    mixer = Mixer(agent_num, embedding_dim, 64)
    total_q = mixer(agent_q, state_embedding)
    assert total_q.shape == (bs, )
    loss = total_q.mean()
    is_differentiable(loss, mixer)


@pytest.mark.unittest
def test_qmix():
    use_mixer = [True, False]
    agent_num, bs, T = 4, 3, 8
    obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
    embedding_dim = 64
    for mix in use_mixer:
        qmix_model = QMix(agent_num, obs_dim, global_obs_dim, action_dim, [128, embedding_dim], mix)
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, agent_num, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, agent_num, action_dim))
            },
            'prev_state': [[None for _ in range(agent_num)] for _ in range(bs)],
            'action': torch.randint(0, action_dim, size=(T, bs, agent_num))
        }
        output = qmix_model(data, single_step=False)
        assert set(output.keys()) == set(['total_q', 'logit', 'next_state', 'action_mask'])
        assert output['total_q'].shape == (T, bs)
        assert output['logit'].shape == (T, bs, agent_num, action_dim)
        assert len(output['next_state']) == bs and all([len(n) == agent_num for n in output['next_state']])
        print(output['next_state'][0][0]['h'].shape)
        loss = output['total_q'].sum()
        is_differentiable(loss, qmix_model)
        data.pop('action')
        output = qmix_model(data, single_step=False)
