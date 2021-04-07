import time
import logging

from nervex.worker import BaseSerialActor
from nervex.envs import get_vec_env_setting, create_env_manager
from nervex.entry.utils import set_pkg_seed
from nervex.data import BufferManager

from nervex.worker.actor.tests.speed_test.fake_policy import FakePolicy
from nervex.worker.actor.tests.speed_test.fake_env import FakeEnv
from nervex.worker.actor.tests.speed_test.test_config import test_config


def compare_test():
    actor_log = logging.getLogger('collector_logger')
    actor_log.disabled = True
    buffer_log = logging.getLogger('agent_buffer_logger')
    buffer_log.disabled = True

    cfg = test_config
    env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    manager_cfg = cfg.env.get('manager', {})
    actor_env = create_env_manager(
        cfg.env.env_manager_type,
        env_fn=FakeEnv,
        env_cfg=actor_env_cfg,
        env_num=len(actor_env_cfg),
        manager_cfg=manager_cfg
    )

    seed = 0
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)

    policy = FakePolicy(cfg.policy)
    actor = BaseSerialActor(cfg.actor)
    replay_buffer = BufferManager(cfg.replay_buffer)
    actor.env = actor_env
    actor.policy = policy.collect_mode

    start = time.time()
    for iter in range(10000):
        if iter % 100 == 0:
            print(iter)
        new_data = actor.generate_data(iter)
        replay_buffer.push(new_data, cur_actor_envstep=iter * 8)
    print('duration', time.time() - start)
    
    actor.close()
    replay_buffer.close()


if __name__ == '__main__':
    compare_test()

