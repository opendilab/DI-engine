import pytest
import torch
from ding.policy.mbpolicy.utils import flatten_batch, unflatten_batch, q_evaluation


@pytest.mark.unittest
def test_flatten_unflattent_batch():
    T, B, C, H, W = 10, 20, 3, 255, 255
    data = torch.randn(T, B, C, H, W)
    data, batch_dim = flatten_batch(data, nonbatch_dims=3)
    assert data.shape == (T * B, C, H, W) and batch_dim == (T, B)
    data = unflatten_batch(data, batch_dim)
    assert data.shape == (T, B, C, H, W)

    T, B, N = 10, 20, 100
    data = torch.randn(T, B, N)
    data, batch_dim = flatten_batch(data, nonbatch_dims=1)
    assert data.shape == (T * B, N) and batch_dim == (T, B)
    data = unflatten_batch(data, batch_dim)
    assert data.shape == (T, B, N)


@pytest.mark.unittest
def test_q_evaluation():
    T, B, O, A = 10, 20, 100, 30
    obss = torch.randn(T, B, O)
    actions = torch.randn(T, B, A)

    def fake_q_fn(obss, actions):
        # obss:    flatten_B * O
        # actions: flatten_B * A
        # return:  flatten_B
        return obss.sum(-1)

    q_value = q_evaluation(obss, actions, fake_q_fn)
    assert q_value.shape == (T, B)


# @pytest.mark.unittest
# def test_rollout():
#     T, B, O, A = 10, 20, 100, 30

#     def fake_policy_fn(obs):
#         return torch.randn(B, A), torch.zeros(B)

#     def fake_env(obs, action):
#         # r, s, done
#         return (torch.zeros(B), torch.rand(B, O), obs.sum(-1) > 0)

#     obs = torch.rand(B, O)
#     obss, actions, rewards, aug_rewards, dones = \
#         rollout(obs, fake_policy_fn, T, fake_env)
#     assert obss.shape == (T + 1, B, O)
#     assert actions.shape == (T + 1, B, A)
#     assert rewards.shape == (T, B)
#     assert aug_rewards.shape == (T + 1, B)
#     assert dones.shape == (T + 1, B)
