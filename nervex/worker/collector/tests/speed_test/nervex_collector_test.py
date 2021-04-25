import time
import logging
from easydict import EasyDict

from nervex.worker import BaseSerialCollector
from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.entry.utils import set_pkg_seed
from nervex.data import BufferManager
from nervex.utils import deep_merge_dicts

from nervex.worker.collector.tests.speed_test.fake_policy import FakePolicy
from nervex.worker.collector.tests.speed_test.fake_env import FakeEnv, env_sum
from nervex.worker.collector.tests.speed_test.test_config import test_config

FAST_MODE = False


def compare_test(cfg, out_str, seed):
    global FAST_MODE
    duration_list = []
    repeat_times = 1 if FAST_MODE else 3
    for i in range(repeat_times):
        cfg = deep_merge_dicts(test_config, cfg)
        cfg.env.collector_env_num = 8
        if cfg.env.env_manager_type == 'async_subprocess':
            cfg.env.manager.wait_num = 8 - 1
        env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
        manager_cfg = cfg.env.get('manager', {})
        collector_env = create_env_manager(
            cfg.env.env_manager_type,
            env_fn=FakeEnv,
            env_cfg=collector_env_cfg,
            env_num=len(collector_env_cfg),
            manager_cfg=manager_cfg
        )
        collector_env.seed(seed)

        policy = FakePolicy(cfg.policy)
        collector = BaseSerialCollector(cfg.collector)
        replay_buffer = BufferManager(cfg.replay_buffer)
        collector.env = collector_env
        collector.policy = policy.collect_mode

        start = time.time()
        iters = 150 if FAST_MODE else 300
        for iter in range(iters):
            if iter % 50 == 0:
                print('\t', iter)
            new_data = collector.collect_data(iter)
            replay_buffer.push(new_data, cur_collector_envstep=iter * 8)
        duration_list.append(time.time() - start)
        print(
            '\tduration: {}, env_sleep: {}, policy_sleep: {}({})'.format(
                time.time() - start, env_sum, policy.policy_sum, policy.policy_times
            )
        )

        collector.close()
        replay_buffer.close()
    print(cfg)
    print('avg duration: {}; ({})'.format(sum(duration_list) / len(duration_list), duration_list))
    out_str.append('avg duration: {}; ({})'.format(sum(duration_list) / len(duration_list), duration_list))


if __name__ == '__main__':
    collector_log = logging.getLogger('collector_logger')
    collector_log.disabled = True
    buffer_log = logging.getLogger('agent_buffer_logger')
    buffer_log.disabled = True

    seed = 0
    set_pkg_seed(seed, use_cuda=False)

    cfgs = [
        dict(
            size="small",
            env=dict(
                obs_dim=64,
                action_dim=2,
                episode_step=500,
                reset_time=0.1,
                step_time=0.005,
            ),
            policy=dict(forward_time=0.004, ),
            actor=dict(n_sample=80, ),
        ),

        # dict(
        #     size="middle",
        #     env=dict(
        #         obs_dim=int(3e2),  # int(3e3),
        #         action_dim=2,
        #         episode_step=500,
        #         reset_time=0.5,  # 2
        #         step_time=0.01,
        #     ),
        #     policy=dict(forward_time=0.008, ),
        #     actor=dict(n_sample=80, ),
        # ),

        # dict(
        #     size="big",
        #     env=dict(
        #         obs_dim=int(3e3),  # int(3e6),
        #         action_dim=2,
        #         episode_step=500,
        #         reset_time=2,
        #         step_time=0.1,
        #     ),
        #     policy=dict(forward_time=0.02, ),
        #     actor=dict(n_sample=80, ),
        # ),
    ]
    out_str = []
    for cfg in cfgs:
        envm_list = ['async_subprocess', 'subprocess']  # ['base', 'async_subprocess', 'subprocess']
        for envm in envm_list:
            reset_list = [1, 5]  # [1, 5]
            for reset_ratio in reset_list:
                cfg = EasyDict(cfg)
                cfg.env.reset_time *= reset_ratio
                cfg.env.env_manager_type = envm
                print('=={}, {}, reset x{}'.format(cfg.size, envm, reset_ratio))
                out_str.append('=={}, {}, reset x{}'.format(cfg.size, envm, reset_ratio))
                compare_test(cfg, out_str, seed)
    print('\n'.join(out_str))
