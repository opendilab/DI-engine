from ditk import logging
import time
import copy
import pytest
import numpy as np
import gym
from easydict import EasyDict
from functools import partial

from ding.worker import SampleSerialCollector, NaiveReplayBuffer
from ding.envs import get_vec_env_setting, create_env_manager, AsyncSubprocessEnvManager, SyncSubprocessEnvManager,\
    BaseEnvManager, get_env_manager_cls, DingEnvWrapper
from ding.utils import deep_merge_dicts, set_pkg_seed, pretty_print

from ding.worker.collector.tests.speed_test.fake_policy import FakePolicy
from ding.worker.collector.tests.speed_test.fake_env import FakeEnv

n_sample = 80
env_policy_cfg_dict = dict(
    # Small env and policy, such as Atari/Mujoco
    small=dict(
        size="small",
        env=dict(
            collector_env_num=8,
            obs_dim=64,
            action_dim=2,
            episode_step=500,
            reset_time=0.1,
            step_time=0.005,
            manager=dict(),
        ),
        policy=dict(forward_time=0.004),
    ),
    # Middle env and policy, such as Carla/Sumo/Vizdoom
    middle=dict(
        size="middle",
        env=dict(
            collector_env_num=8,
            obs_dim=int(3e2),  # int(3e3),
            action_dim=2,
            episode_step=500,
            reset_time=0.5,
            step_time=0.01,
            manager=dict(),
        ),
        policy=dict(forward_time=0.008),
    ),
    # Big env and policy, such as SC2 full game
    big=dict(
        size="big",
        env=dict(
            collector_env_num=8,
            obs_dim=int(3e3),  # int(3e6),
            action_dim=2,
            episode_step=500,
            reset_time=2,
            step_time=0.1,
            manager=dict(),
        ),
        policy=dict(forward_time=0.02)
    ),
    # cartpole env
    cartpole=dict(
        size='cartpole',
        env=dict(collector_env_num=8, stop_value=195, reset_time=0.5, manager=dict(reset_inplace=True, )),
        policy=dict(
            cuda=False,
            model=dict(
                obs_shape=4,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            collect=dict(
                n_sample=n_sample,
                collector=dict(collect_print_freq=1000000),
            ),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )
)


def wrapped_cartpole_env():
    return DingEnvWrapper(gym.make('CartPole-v0'))


def wrapped_gym_cartpole_env():
    return gym.make('CartPole-v0')


# SLOW MODE: used in normal test
#   - Repeat 3 times; Collect 300 times;
#   - Test on small + middle + big env
#   - Test on base + async_subprocess + sync_subprocess env manager
#   - Test with reset_ratio = 1 and 5.
# FAST MODE: used in CI benchmark test
#   - Only once (No repeat); Collect 50 times;
#   - Test on small env
#   - Test on base + sync_subprocess env manager
#   - Test with reset_ratio = 1.
FAST_MODE = True
if FAST_MODE:
    # Note: 'base' takes approximately 6 times longer than 'subprocess'
    test_env_manager_list = ['base', 'subprocess', 'gym_vector']
    test_env_policy_cfg_dict = {'small': env_policy_cfg_dict['small'], 'cartpole': env_policy_cfg_dict['cartpole']}
    env_reset_ratio_list = [1]
    repeat_times_per_test = 1
    collect_times_per_repeat = 50
else:
    test_env_manager_list = ['base', 'subprocess', 'sync_subprocess', 'gym_vector']
    test_env_policy_cfg_dict = env_policy_cfg_dict
    env_reset_ratio_list = [1, 5]
    repeat_times_per_test = 3
    collect_times_per_repeat = 300


def compare_test(cfg: EasyDict, seed: int, test_name: str) -> None:
    duration_list = []
    total_collected_sample = n_sample * collect_times_per_repeat
    for i in range(repeat_times_per_test):
        # create collector_env
        collector_env_cfg = copy.deepcopy(cfg.env)
        collector_env_num = collector_env_cfg.collector_env_num
        if cfg.size == 'cartpole':
            if cfg.env.manager.type == 'gym_vector':
                collector_env_fns = [wrapped_gym_cartpole_env for _ in range(collector_env_num)]
            else:
                collector_env_fns = [wrapped_cartpole_env for _ in range(collector_env_num)]
        else:
            collector_env_fns = [partial(FakeEnv, cfg=collector_env_cfg) for _ in range(collector_env_num)]

        collector_env = create_env_manager(cfg.env.manager, collector_env_fns)
        collector_env.seed(seed)
        # create policy
        policy = FakePolicy(cfg.policy)

        # create collector and buffer
        collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)
        replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer)

        # collect test

        t1 = time.time()
        for i in range(collect_times_per_repeat):
            new_data = collector.collect()
            assert len(new_data) == n_sample
            replay_buffer.push(new_data, cur_collector_envstep=i * n_sample)
        duration_list.append(time.time() - t1)

        # close and release
        collector.close()
        replay_buffer.close()
        del policy
        del collector
        del replay_buffer

    fps = [total_collected_sample / duration for duration in duration_list]

    template = "Test Name: {}\t Test Result: Avg FPS(env frame per second): {:.3f}±{:.3f} frame/s"
    print(template.format(test_name, np.mean(fps), np.std(fps)))


@pytest.mark.benchmark
def test_collector_profile():
    # ignore them for clear log
    collector_log = logging.getLogger('collector_logger')
    collector_log.disabled = True
    buffer_log = logging.getLogger('buffer_logger')
    buffer_log.disabled = True

    seed = 0
    set_pkg_seed(seed, use_cuda=False)
    print("=========== test_collector_profile ===========")

    for cfg_name, env_policy_cfg in test_env_policy_cfg_dict.items():
        for env_manager_type in test_env_manager_list:
            for env_reset_ratio in env_reset_ratio_list:

                test_name = '{}-{}-reset{}'.format(cfg_name, env_manager_type, env_reset_ratio)
                copy_cfg = EasyDict(copy.deepcopy(env_policy_cfg))
                env_manager_cfg = EasyDict({'type': env_manager_type})

                # modify args inplace
                copy_cfg.policy = deep_merge_dicts(FakePolicy.default_config(), copy_cfg.policy)
                copy_cfg.policy.collect.collector = deep_merge_dicts(
                    SampleSerialCollector.default_config(), copy_cfg.policy.collect.collector
                )
                copy_cfg.policy.collect.collector.n_sample = n_sample
                copy_cfg.policy.other.replay_buffer = deep_merge_dicts(
                    NaiveReplayBuffer.default_config(), copy_cfg.policy.other.replay_buffer
                )
                copy_cfg.env.reset_time *= env_reset_ratio
                manager_cfg = get_env_manager_cls(env_manager_cfg).default_config()
                copy_cfg.env.manager = deep_merge_dicts(manager_cfg, copy_cfg.env.manager)
                copy_cfg.env.manager.type = env_manager_type

                compare_test(copy_cfg, seed, test_name)
