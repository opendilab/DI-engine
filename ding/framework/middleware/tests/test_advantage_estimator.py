import pytest

from ding.data.buffer import DequeBuffer
from ding.data import Buffer
from easydict import EasyDict
from ding.framework import OnlineRLContext
import treetensor
import torch
import copy

from ding.framework.middleware.functional.advantage_estimator import gae_estimator
from ding.framework.middleware.functional.advantage_estimator import montecarlo_return_estimator
from ding.utils.data import ttorch_collate

from typing import Any, List, Dict, Optional

from unittest.mock import Mock, patch


class TheModelClass:

    def forward(self, obs: Dict, mode: str) -> Dict:
        return {'value': torch.distributions.uniform.Uniform(0, 4).sample([len(obs.data)])}


class MockPolicy(Mock):

    def __init__(self, model) -> None:
        super(MockPolicy, self).__init__()
        self._model = model

    def get_attribute(self, name: str) -> Any:
        return self._model


def call_gae_estimator(batch_size: int = 32, trajectory_end_idx_size: int = 5, buffer: Optional[Buffer] = None):
    cfg = EasyDict(
        {
            'policy': {
                'model': {
                    'obs_shape': 4,
                    'action_shape': 2,
                },
                'collect': {
                    'discount_factor': 0.9,
                    'gae_lambda': 0.95
                },
                'cuda': False
            }
        }
    )

    ctx = OnlineRLContext()
    assert trajectory_end_idx_size <= batch_size

    ctx.trajectory_end_idx = treetensor.torch.randint(low=0, high=batch_size, size=(trajectory_end_idx_size, ))
    ctx.trajectories = [
        treetensor.torch.Tensor(
            {
                'action': treetensor.torch.randint(low=0, high=2, size=(1, )),
                'collect_train_iter': [0],
                'done': False,
                'logit': treetensor.torch.randn(2),
                'next_obs': treetensor.torch.randn(4),
                'obs': treetensor.torch.randn(4),
                'reward': [1.0],
                'value': torch.distributions.uniform.Uniform(0, 4).sample([1])
            }
        ) for _ in range(batch_size)
    ]
    ctx.trajectories_copy = ttorch_collate(copy.deepcopy(ctx.trajectories), cat_1dim=True)
    traj_flag = ctx.trajectories_copy.done.clone()
    traj_flag[ctx.trajectory_end_idx] = True
    ctx.trajectories_copy.traj_flag = traj_flag

    with patch("ding.policy.Policy", MockPolicy):
        gae_estimator(cfg, MockPolicy(TheModelClass()), buffer)(ctx)

    if buffer is not None:
        train_data = [d.data for d in list(buffer.storage)]
        for d in train_data:
            d.logit = d.logit
            d.next_obs = d.next_obs
            d.obs = d.obs
        ctx.train_data = ttorch_collate(train_data, cat_1dim=True)

    assert ctx.trajectories is None
    assert torch.equal(ctx.trajectories_copy.action, ctx.train_data.action)
    assert torch.equal(ctx.trajectories_copy.collect_train_iter, ctx.train_data.collect_train_iter)
    assert torch.equal(ctx.trajectories_copy.logit, ctx.train_data.logit)
    assert torch.equal(ctx.trajectories_copy.next_obs, ctx.train_data.next_obs)
    assert torch.equal(ctx.trajectories_copy.obs, ctx.train_data.obs)
    assert torch.equal(ctx.trajectories_copy.reward, ctx.train_data.reward)
    assert torch.equal(ctx.trajectories_copy.traj_flag, ctx.train_data.traj_flag)


@pytest.mark.unittest
def test_gae_estimator():
    batch_size = 32
    trajectory_end_idx_size = 5
    call_gae_estimator(batch_size, trajectory_end_idx_size)
    call_gae_estimator(batch_size, trajectory_end_idx_size, DequeBuffer(size=batch_size))


class MockPGPolicy(Mock):

    def __init__(self, cfg) -> None:
        super(MockPGPolicy, self).__init__()
        self._cfg = EasyDict(cfg)
        self._gamma = self._cfg.collect.discount_factor
        self._unroll_len = self._cfg.collect.unroll_len

    def get_attribute(self, name: str) -> Any:
        return self._model


def call_montecarlo_return_estimator(batch_size: int = 32):

    cfg = dict(
        learn=dict(ignore_done=False, ),
        collect=dict(
            unroll_len=1,
            discount_factor=0.9,
        ),
    )
    ctx = OnlineRLContext()
    ctx.episodes = [
        [
            treetensor.torch.Tensor(
                {
                    'action': treetensor.torch.randint(low=0, high=2, size=(1, )),
                    'collect_train_iter': [0],
                    'done': False if i != batch_size - 1 else True,
                    'logit': treetensor.torch.randn(2),
                    'next_obs': treetensor.torch.randn(4),
                    'obs': treetensor.torch.randn(4),
                    'reward': [1.0],
                    'value': torch.distributions.uniform.Uniform(0, 4).sample([1])
                }
            ) for i in range(batch_size)
        ]
    ]
    ctx.episodes_copy = treetensor.torch.concat(
        [ttorch_collate(copy.deepcopy(episode), cat_1dim=True) for episode in ctx.episodes], dim=0
    )
    with patch("ding.policy.Policy", MockPGPolicy):
        montecarlo_return_estimator(MockPGPolicy(cfg))(ctx)

    assert torch.equal(ctx.episodes_copy.action, ctx.train_data.action)
    assert torch.equal(ctx.episodes_copy.collect_train_iter, ctx.train_data.collect_train_iter)
    assert torch.equal(ctx.episodes_copy.logit, ctx.train_data.logit)
    assert torch.equal(ctx.episodes_copy.next_obs, ctx.train_data.next_obs)
    assert torch.equal(ctx.episodes_copy.obs, ctx.train_data.obs)
    assert torch.equal(ctx.episodes_copy.reward, ctx.train_data.reward)


@pytest.mark.unittest
def test_montecarlo_return_estimator():
    batch_size = 32
    call_montecarlo_return_estimator(batch_size)
