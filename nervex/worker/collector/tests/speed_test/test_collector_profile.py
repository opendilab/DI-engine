import time
import logging
from easydict import EasyDict
import pytest
from functools import partial
import copy

from nervex.worker import BaseSerialCollector
from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.entry.utils import set_pkg_seed
from nervex.data import BufferManager
from nervex.utils import deep_merge_dicts

from nervex.worker.collector.tests.speed_test.fake_policy import FakePolicy
from nervex.worker.collector.tests.speed_test.fake_env import FakeEnv, env_sum
from nervex.worker.collector.tests.speed_test.test_config import test_config

# SLOW MODE: Repeat 3 times; Test on small+middle+big env, base+asynnc_subprocess+sync_subprocess env manager.
# FAST MODE: Only once (No repeat); Test on small+middle env, async_subprocess+sync_subprocess env manager.
FAST_MODE = True


def compare_test(cfg, out_str, seed):
    global FAST_MODE
    duration_list = []
    repeat_times = 1 if FAST_MODE else 3
    for i in range(repeat_times):
        env_fn = FakeEnv
        collector_env_cfg = [cfg.env.env_kwargs for _ in range(cfg.env.env_kwargs.collector_env_num)]
        collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
        collector_env.seed(seed)

        policy = FakePolicy(cfg.policy)
        collector = BaseSerialCollector(cfg.collector)
        replay_buffer = BufferManager(cfg.replay_buffer)
        collector.env = collector_env
        collector.policy = policy.collect_mode

        start = time.time()
        iters = 300  # 100 if FAST_MODE else 300
        for iter in range(iters):
            if iter % 50 == 0:
                print('\t', iter)
            new_data = collector.collect_data(iter)
            replay_buffer.push(new_data, cur_collector_envstep=iter * 8)
        duration_list.append(time.time() - start)
        print('\tduration: {}'.format(time.time() - start))
        collector.close()
        replay_buffer.close()
        del policy
        del collector
        del replay_buffer
    print('avg duration: {}; ({})'.format(sum(duration_list) / len(duration_list), duration_list))
    out_str.append('avg duration: {}; ({})'.format(sum(duration_list) / len(duration_list), duration_list))


@pytest.mark.benchmark
def test_collector_profile():
    global FAST_MODE

    collector_log = logging.getLogger('collector_logger')
    collector_log.disabled = True
    buffer_log = logging.getLogger('agent_buffer_logger')
    buffer_log.disabled = True

    seed = 0
    set_pkg_seed(seed, use_cuda=False)

    cfgs = [
        dict(
            size="small",
            env=dict(env_kwargs=dict(
                obs_dim=64,
                action_dim=2,
                episode_step=500,
                reset_time=0.1,
                step_time=0.005,
            ), ),
            policy=dict(forward_time=0.004, ),
            actor=dict(n_sample=80, ),
        ),
        dict(
            size="middle",
            env=dict(
                env_kwargs=dict(
                    obs_dim=int(3e2),  # int(3e3),
                    action_dim=2,
                    episode_step=500,
                    reset_time=0.5,
                    step_time=0.01,
                ),
            ),
            policy=dict(forward_time=0.008, ),
            actor=dict(n_sample=80, ),
        ),

        # Big env(45min) takes much longer time than small(5min) and middle(10min).
        dict(
            size="big",
            env=dict(
                env_kwargs=dict(
                    obs_dim=int(3e3),  # int(3e6),
                    action_dim=2,
                    episode_step=500,
                    reset_time=2,
                    step_time=0.1,
                ),
            ),
            policy=dict(forward_time=0.02, ),
            actor=dict(n_sample=80, ),
        ),
    ]
    out_str = []
    if FAST_MODE:
        cfgs.pop(-1)
    for cfg in cfgs:
        # Note: 'base' takes approximately 6 times longer than 'subprocess'
        if FAST_MODE:
            envm_list = ['async_subprocess', 'subprocess']
        else:
            envm_list = ['base', 'async_subprocess', 'subprocess']
        for envm in envm_list:
            reset_list = [1, 5]  # [1, 5]
            for reset_ratio in reset_list:
                copy_cfg = copy.deepcopy(cfg)
                copy_test_config = copy.deepcopy(test_config)
                copy_cfg = EasyDict(copy_cfg)
                copy_cfg = deep_merge_dicts(copy_test_config, copy_cfg)
                copy_cfg.env.env_kwargs.reset_time *= reset_ratio
                copy_cfg.env.manager.type = envm
                if copy_cfg.env.manager.type == 'base':
                    copy_cfg.env.manager.pop('step_wait_timeout')
                    copy_cfg.env.manager.pop('wait_num')

                print('=={}, {}, reset x{}'.format(copy_cfg.size, envm, reset_ratio))
                print(copy_cfg)
                out_str.append('=={}, {}, reset x{}'.format(copy_cfg.size, envm, reset_ratio))
                compare_test(copy_cfg, out_str, seed)
    print('\n'.join(out_str))
