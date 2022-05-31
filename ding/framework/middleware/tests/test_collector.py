import pytest
import torch
import copy
from unittest.mock import patch
from ding.framework import OnlineRLContext, task
from ding.framework.middleware import TransitionList, inferencer, rolloutor
from ding.framework.middleware import StepCollector, EpisodeCollector
from ding.framework.middleware.tests import MockPolicy, MockEnv, CONFIG


@pytest.mark.unittest
def test_inferencer():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        policy = MockPolicy()
        env = MockEnv()
        inferencer(cfg, policy, env)(ctx)
    assert isinstance(ctx.inference_output, dict)
    assert ctx.inference_output[0] == {'action': torch.Tensor([0.])}  # sum of zeros([2, 2])
    assert ctx.inference_output[1] == {'action': torch.Tensor([4.])}  # sum of ones([2, 2])


@pytest.mark.unittest
def test_rolloutor():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()
    transitions = TransitionList(2)
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        policy = MockPolicy()
        env = MockEnv()
        for _ in range(10):
            inferencer(cfg, policy, env)(ctx)
            rolloutor(cfg, policy, env, transitions)(ctx)
    assert ctx.env_episode == 20  # 10 * env_num
    assert ctx.env_step == 20  # 10 * env_num


@pytest.mark.unittest
def test_step_collector():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    # test no random_collect_size
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        with task.start():
            policy = MockPolicy()
            env = MockEnv()
            collector = StepCollector(cfg, policy, env)
            collector(ctx)
    assert len(ctx.trajectories) == 16
    assert ctx.trajectory_end_idx == [7, 15]

    # test with random_collect_size
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        with task.start():
            policy = MockPolicy()
            env = MockEnv()
            collector = StepCollector(cfg, policy, env, random_collect_size=8)
            collector(ctx)
    assert len(ctx.trajectories) == 16
    assert ctx.trajectory_end_idx == [7, 15]


@pytest.mark.unittest
def test_episode_collector():
    cfg = copy.deepcopy(CONFIG)
    ctx = OnlineRLContext()

    # test no random_collect_size
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        with task.start():
            policy = MockPolicy()
            env = MockEnv()
            collector = EpisodeCollector(cfg, policy, env)
            collector(ctx)
    assert len(ctx.episodes) == 16

    # test with random_collect_size
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        with task.start():
            policy = MockPolicy()
            env = MockEnv()
            collector = EpisodeCollector(cfg, policy, env, random_collect_size=8)
            collector(ctx)
    assert len(ctx.episodes) == 16
