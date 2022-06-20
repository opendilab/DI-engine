import pytest
import torch
import copy
from unittest.mock import patch
from ding.framework import OnlineRLContext, task
from ding.framework.middleware import TransitionList, inferencer, rolloutor
from ding.framework.middleware import StepCollector, EpisodeCollector
from ding.framework.middleware.tests import MockPolicy, MockEnv, CONFIG
from ding.framework.middleware import BattleTransitionList
from easydict import EasyDict
import copy


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


@pytest.mark.unittest
def test_battle_transition_list():
    env_num = 2
    unroll_len = 32
    transition_list = BattleTransitionList(env_num, unroll_len)
    len_env_0 = 48
    len_env_1 = 72

    for i in range(len_env_0):
        timestep = EasyDict({'obs': i, 'done': False})
        transition_list.append(env_id=0, transition=timestep)

    transition_list.append(env_id=0, transition=EasyDict({'obs': len_env_0, 'done': True}))

    for i in range(len_env_1):
        timestep = EasyDict({'obs': i, 'done': False})
        transition_list.append(env_id=1, transition=timestep)

    transition_list.append(env_id=1, transition=EasyDict({'obs': len_env_1, 'done': True}))

    len_env_0_2 = 12
    len_env_1_2 = 72

    for i in range(len_env_0_2):
        timestep = EasyDict({'obs': i, 'done': False})
        transition_list.append(env_id=0, transition=timestep)

    transition_list.append(env_id=0, transition=EasyDict({'obs': len_env_0_2, 'done': True}))

    for i in range(len_env_1_2):
        timestep = EasyDict({'obs': i, 'done': False})
        transition_list.append(env_id=1, transition=timestep)

    transition_list_2 = copy.deepcopy(transition_list)

    env_0_result = transition_list.get_trajectories(env_id=0)
    env_1_result = transition_list.get_trajectories(env_id=1)

    # print(env_0_result)
    # print(env_1_result)

    # print(env_0_result)
    assert len(env_0_result) == 3
    # print(env_1_result)
    assert len(env_1_result) == 4

    for trajectory in env_0_result:
        assert len(trajectory) == unroll_len
    for trajectory in env_1_result:
        assert len(trajectory) == unroll_len
    #env_0
    i = 0
    trajectory = env_0_result[0]
    for transition in trajectory:
        assert transition.obs == i
        i += 1

    trajectory = env_0_result[1]
    i = len_env_0 - unroll_len + 1
    for transition in trajectory:
        assert transition.obs == i
        i += 1

    trajectory = env_0_result[2]
    test_number = 0
    for i, transition in enumerate(trajectory):
        if i < unroll_len - len_env_0_2 - 1:
            assert transition.obs == 0
        else:
            assert transition.obs == test_number
            test_number += 1

    #env_1
    i = 0
    for trajectory in env_1_result[:2]:
        assert len(trajectory) == unroll_len
        for transition in trajectory:
            assert transition.obs == i
            i += 1

    trajectory = env_1_result[2]
    assert len(trajectory) == unroll_len

    i = len_env_1 - unroll_len + 1
    for transition in trajectory:
        assert transition.obs == i
        i += 1

    trajectory = env_1_result[3]
    assert len(trajectory) == unroll_len
    i = 0
    for transition in trajectory:
        assert transition.obs == i
        i += 1

    # print(env_0_result)
    # print(env_1_result)
    # print(transition_list._transitions[0])
    # print(transition_list._transitions[1])

    transition_list_2.clear_newest_episode(env_id=0)
    transition_list_2.clear_newest_episode(env_id=1)

    assert len(transition_list_2._transitions[0]) == 2
    assert len(transition_list_2._transitions[1]) == 1


if __name__ == '__main__':
    test_battle_transition_list()