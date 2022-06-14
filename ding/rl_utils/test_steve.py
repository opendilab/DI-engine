import pytest
import torch
from .steve import steve_target


@pytest.mark.unittest
def test_steve_target():
    B = 2
    M, N, L = 3, 4, 5
    obs_tplus1 = torch.randn(B, 16)
    reward_t = torch.randn(B, )
    done_t = torch.randint(0, 2, size=(B, )).float()

    def q_fn(obs, action):
        return torch.randn(obs.shape[0], M)

    def rew_fn(obs, action, next_obs):
        return torch.randn(obs.shape[0], N)

    def policy_fn(obs):
        return torch.randn(obs.shape[0], )

    def transition_fn(obs, action):
        return torch.randn(obs.shape[0], L, obs.shape[1])

    def done_fn(obs, action, next_obs):
        return torch.randint(0, 2, size=(obs.shape[0], L)).float()

    return_ = steve_target(
        obs_tplus1,
        reward_t,
        done_t,
        q_fn,
        rew_fn,
        policy_fn,
        transition_fn,
        done_fn,
        rollout_step=6,
        discount_factor=0.99,
        ensemble_num=(M, N, L)
    )
    assert return_.shape == (B, )
