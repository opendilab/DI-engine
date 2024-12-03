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
    ctx = OnlineRLContext()
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        policy = MockPolicy()
        env = MockEnv()
        inferencer(0, policy, env)(ctx)
    assert isinstance(ctx.inference_output, dict)
    assert ctx.inference_output[0] == {'action': torch.Tensor([0.])}  # sum of zeros([2, 2])
    assert ctx.inference_output[1] == {'action': torch.Tensor([4.])}  # sum of ones([2, 2])


@pytest.mark.unittest
def test_rolloutor():
    N = 20
    ctx = OnlineRLContext()
    transitions = TransitionList(2)
    with patch("ding.policy.Policy", MockPolicy), patch("ding.envs.BaseEnvManagerV2", MockEnv):
        policy = MockPolicy()
        env = MockEnv()
        i = inferencer(0, policy, env)
        r = rolloutor(policy, env, transitions)
        for _ in range(N):
            i(ctx)
            r(ctx)
    assert ctx.env_step == N * 2  # N * env_num
    assert ctx.env_episode >= N // 10 * 2  # N * env_num


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
