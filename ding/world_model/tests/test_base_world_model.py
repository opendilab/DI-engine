import pytest
import torch
from easydict import EasyDict
from ding.world_model.base_world_model import DreamWorldModel


@pytest.mark.unittest
def test_rollout():
    fake_config = EasyDict(
        train_freq=250,  # w.r.t environment step
        eval_freq=20,  # w.r.t environment step
        cuda=False,
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=20000,
            rollout_end_step=150000,
            rollout_length_min=1,
            rollout_length_max=25,
        )
    )
    envstep = 150000
    T, B, O, A = 25, 20, 100, 30

    class FakeModel(DreamWorldModel):

        def train(self, env_buffer, envstep, train_iter):
            pass

        def eval(self, env_buffer, envstep, train_iter):
            pass

        def step(self, obs, action):
            # r, s, done
            return (torch.zeros(B), torch.rand(B, O), obs.sum(-1) > 0)

    def fake_policy_fn(obs):
        return torch.randn(B, A), torch.zeros(B)

    fake_model = FakeModel(fake_config, None, None)

    obs = torch.rand(B, O)
    obss, actions, rewards, aug_rewards, dones = \
        fake_model.rollout(obs, fake_policy_fn, envstep)
    assert obss.shape == (T + 1, B, O)
    assert actions.shape == (T + 1, B, A)
    assert rewards.shape == (T, B)
    assert aug_rewards.shape == (T + 1, B)
    assert dones.shape == (T + 1, B)
