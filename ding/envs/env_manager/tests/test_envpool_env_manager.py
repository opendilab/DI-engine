import time
import pytest
import numpy as np
from easydict import EasyDict

from ..envpool_env_manager import PoolEnvManager

env_num_args = [[16, 8], [8, 8]]


@pytest.mark.envpooltest
@pytest.mark.parametrize('env_num, batch_size', env_num_args)
class TestPoolEnvManager:

    def test_naive(self, env_num, batch_size):
        env_manager_cfg = EasyDict(
            {
                'env_id': 'Pong-v5',
                'env_num': env_num,
                'batch_size': batch_size,
                'seed': 3,
                # env wrappers
                'episodic_life': False,
                'reward_clip': False,
                'gray_scale': True,
                'stack_num': 4,
                'frame_skip': 4,
            }
        )
        env_manager = PoolEnvManager(env_manager_cfg)
        assert env_manager._closed
        env_manager.launch()
        # Test step
        start_time = time.time()
        for count in range(20):
            env_id = env_manager.ready_obs.keys()
            action = {i: np.random.randint(4) for i in env_id}
            timestep = env_manager.step(action)
            assert len(timestep) == env_manager_cfg.batch_size
            print('Count {}'.format(count))
            print([v.info for v in timestep.values()])
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))
        # Test close
        env_manager.close()
        assert env_manager._closed
