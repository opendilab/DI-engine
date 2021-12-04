import logging
import time
import copy
import pytest
import numpy as np
from easydict import EasyDict
from functools import partial

from ding.worker import SampleSerialCollector, NaiveReplayBuffer
from ding.envs import get_vec_env_setting, create_env_manager, AsyncSubprocessEnvManager, SyncSubprocessEnvManager,\
    BaseEnvManager, get_env_manager_cls
from ding.utils import deep_merge_dicts, set_pkg_seed, pretty_print

from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
import gym
from ding.rl_utils import get_epsilon_greedy_fn


# Get DI-engine form env class
def wrapped_cartpole_env():
    return DingEnvWrapper(gym.make('CartPole-v0'))


env_policy_cfg_dict = dict(
    cartpole_dqn=dict(
        size='cartpole_dqn',
        env=dict(
            collector_env_num=8,
            stop_value=195,
            reset_time=0.5,
        ),
        policy=dict(
            cuda=False,
            model=dict(
                obs_shape=4,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            collect=dict(
                n_sample=80,
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
    #test_env_manager_list = ['base', 'subprocess', 'gym3_async']
    test_env_manager_list = ['base', 'subprocess', 'async_subprocess', 'gym_async']
    test_env_policy_cfg_dict = {'cartpole_dqn': env_policy_cfg_dict['cartpole_dqn']}
    env_reset_ratio_list = [1]
    repeat_times_per_test = 1
    collect_times_per_repeat = 50
    n_sample = 80
else:
    test_env_manager_list = ['base', 'subprocess', 'sync_subprocess']
    test_env_policy_cfg_dict = env_policy_cfg_dict
    env_reset_ratio_list = [1, 5]
    repeat_times_per_test = 3
    collect_times_per_repeat = 300
    n_sample = 80


def compare_test(cfg: EasyDict, seed: int, test_name: str) -> None:
    print('=' * 100 + '\nTest Name: {}\nCfg:'.format(test_name))
    #pretty_print(cfg)

    duration_list = []
    total_collected_sample = n_sample * collect_times_per_repeat
    for i in range(repeat_times_per_test):
        # create collector_env
        collector_env_cfg = copy.deepcopy(cfg.env)
        collector_env_num = collector_env_cfg.collector_env_num
        collector_env_fns = [wrapped_cartpole_env for _ in range(collector_env_num)]

        collector_env = create_env_manager(cfg.env.manager, collector_env_fns)
        collector_env.seed(seed)
        # create policy
        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)

        # create collector and buffer
        collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode)
        replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer)

        # collect test
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
        eps = epsilon_greedy(collector.envstep)

        t1 = time.time()
        for i in range(collect_times_per_repeat):
            new_data = collector.collect(policy_kwargs={'eps': eps})
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
    print('\nTest Result:\nAvg FPS(env frame per second): {:.3f}±{:.3f} frame/s'.format(np.mean(fps), np.std(fps)))
    print('=' * 100)


@pytest.mark.benchmark
def test_collector_profile():
    # ignore them for clear log
    collector_log = logging.getLogger('collector_logger')
    collector_log.disabled = True
    buffer_log = logging.getLogger('buffer_logger')
    buffer_log.disabled = True

    seed = 0
    set_pkg_seed(seed, use_cuda=False)

    for cfg_name, env_policy_cfg in test_env_policy_cfg_dict.items():
        for env_manager_type in test_env_manager_list:
            for env_reset_ratio in env_reset_ratio_list:
                print(env_policy_cfg)
                test_name = '{}-{}-reset{}'.format(cfg_name, env_manager_type, env_reset_ratio)
                copy_cfg = EasyDict(copy.deepcopy(env_policy_cfg))
                env_manager_cfg = EasyDict({'type': env_manager_type})

                # modify args inplace
                copy_cfg.policy = deep_merge_dicts(DQNPolicy.default_config(), copy_cfg.policy)
                copy_cfg.policy.collect.collector = deep_merge_dicts(
                    SampleSerialCollector.default_config(), copy_cfg.policy.collect.collector
                )
                copy_cfg.policy.collect.collector.n_sample = n_sample
                copy_cfg.policy.other.replay_buffer = deep_merge_dicts(
                    NaiveReplayBuffer.default_config(), copy_cfg.policy.other.replay_buffer
                )
                copy_cfg.env.reset_time *= env_reset_ratio
                copy_cfg.env.manager = get_env_manager_cls(env_manager_cfg).default_config()
                copy_cfg.env.manager.type = env_manager_type

                compare_test(copy_cfg, seed, test_name)