import pytest
import random
import torch
import copy
from unittest.mock import Mock, patch
from ding.data.buffer import DequeBuffer
from ding.framework import OnlineRLContext, task
from ding.framework.middleware import trainer, multistep_trainer, OffPolicyLearner, HERLearner
from ding.framework.middleware.tests import MockHerRewardModel, CONFIG


class MockPolicy(Mock):
    # MockPolicy class for train mode
    def forward(self, train_data, **kwargs):
        res = {
            'total_loss': 0.1,
        }
        return res


class MultiStepMockPolicy(Mock):
    # MockPolicy class for multi-step train mode
    def forward(self, train_data, **kwargs):
        res = [
            {
                'total_loss': 0.1,
            },
            {
                'total_loss': 1.0,
            },
        ]
        return res


def get_mock_train_input():
    data = {'obs': torch.rand(2, 2), 'next_obs': torch.rand(2, 2), 'reward': random.random(), 'info': {}}
    return data


@pytest.mark.unittest
def test_trainer():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    ctx.train_data = None
    with patch("ding.policy.Policy", MockPolicy):
        policy = MockPolicy()
        for _ in range(10):
            trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 0

    ctx.train_data = get_mock_train_input()
    with patch("ding.policy.Policy", MockPolicy):
        policy = MockPolicy()
        for _ in range(30):
            trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 30
    assert ctx.train_output["total_loss"] == 0.1


@pytest.mark.unittest
def test_multistep_trainer():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    ctx.train_data = None
    with patch("ding.policy.Policy", MockPolicy):
        policy = MockPolicy()
        for _ in range(10):
            trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 0

    ctx.train_data = get_mock_train_input()
    with patch("ding.policy.Policy", MultiStepMockPolicy):
        policy = MultiStepMockPolicy()
        for _ in range(30):
            multistep_trainer(cfg, policy)(ctx)
    assert ctx.train_iter == 60
    assert ctx.train_output[0]["total_loss"] == 0.1
    assert ctx.train_output[1]["total_loss"] == 1.0


@pytest.mark.unittest
def test_offpolicy_learner():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()
    buffer = DequeBuffer(size=10)
    for _ in range(10):
        buffer.push(get_mock_train_input())
    with patch("ding.policy.Policy", MockPolicy):
        with task.start():
            policy = MockPolicy()
            learner = OffPolicyLearner(cfg, policy, buffer)
            learner(ctx)
    assert len(ctx.train_output) == 4


@pytest.mark.unittest
def test_her_learner():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()
    buffer = DequeBuffer(size=10)
    for _ in range(10):
        buffer.push([get_mock_train_input(), get_mock_train_input()])
    with patch("ding.policy.Policy", MockPolicy), patch("ding.reward_model.HerRewardModel", MockHerRewardModel):
        with task.start():
            policy = MockPolicy()
            her_reward_model = MockHerRewardModel()
            learner = HERLearner(cfg, policy, buffer, her_reward_model)
            learner(ctx)
    assert len(ctx.train_output) == 4
