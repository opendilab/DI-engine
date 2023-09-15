import pytest
import torch
from ding.torch_utils import is_differentiable
from ding.model.template import HAVAC


@pytest.mark.unittest
class TestHAVAC:

    def test_havac_rnn_actor(self):
        # discrete+rnn
        bs, T = 3, 8
        obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, action_dim))
            },
            'actor_prev_state': [None for _ in range(bs)],
        }
        model = HAVAC(
            agent_obs_shape=obs_dim,
            global_obs_shape=global_obs_dim,
            action_shape=action_dim,
            use_lstm=True,
        )
        output = model(data, mode='compute_actor')
        assert set(output.keys()) == set(['logit', 'actor_next_state', 'hidden_state'])
        assert output['logit'].shape == (T, bs, action_dim)
        assert len(output['actor_next_state']) == bs
        print(output['actor_next_state'][0]['h'].shape)
        loss = output['logit'].sum()
        is_differentiable(loss, model.actor)

    def test_havac_rnn_critic(self):
        # discrete+rnn
        bs, T = 3, 8
        obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, action_dim))
            },
            'critic_prev_state': [None for _ in range(bs)],
        }
        model = HAVAC(
            agent_obs_shape=obs_dim,
            global_obs_shape=global_obs_dim,
            action_shape=action_dim,
            use_lstm=True,
        )
        output = model(data, mode='compute_critic')
        assert set(output.keys()) == set(['value', 'critic_next_state', 'hidden_state'])
        assert output['value'].shape == (T, bs)
        assert len(output['critic_next_state']) == bs
        print(output['critic_next_state'][0]['h'].shape)
        loss = output['value'].sum()
        is_differentiable(loss, model.critic)

    def test_havac_rnn_actor_critic(self):
        # discrete+rnn
        bs, T = 3, 8
        obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
        data = {
            'obs': {
                'agent_state': torch.randn(T, bs, obs_dim),
                'global_state': torch.randn(T, bs, global_obs_dim),
                'action_mask': torch.randint(0, 2, size=(T, bs, action_dim))
            },
            'actor_prev_state': [None for _ in range(bs)],
            'critic_prev_state': [None for _ in range(bs)],
        }
        model = HAVAC(
            agent_obs_shape=obs_dim,
            global_obs_shape=global_obs_dim,
            action_shape=action_dim,
            use_lstm=True,
        )
        output = model(data, mode='compute_actor_critic')
        assert output['logit'].shape == (T, bs, action_dim)
        assert output['value'].shape == (T, bs)
        loss = output['logit'].sum() + output['value'].sum()
        is_differentiable(loss, model)


# test_havac_rnn_actor()
# test_havac_rnn_critic()
# test_havac_rnn_actor_critic()
