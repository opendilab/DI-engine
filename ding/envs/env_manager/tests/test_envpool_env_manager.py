import time
import pytest
import numpy as np
from easydict import EasyDict

from ding.envs.env_manager.envpool_env_manager import PoolEnvManager

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
        for count in range(5):
            env_id = env_manager.ready_obs.keys()
            action = {i: np.random.randint(4) for i in env_id}
            timestep = env_manager.step(action)
            assert len(timestep) == env_manager_cfg.batch_size
        env_manager.close()
        assert env_manager._closed
