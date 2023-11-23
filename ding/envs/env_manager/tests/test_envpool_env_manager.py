import time
import pytest
import numpy as np
from easydict import EasyDict

from ding.envs.env_manager.envpool_env_manager import PoolEnvManager, PoolEnvManagerV2

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


@pytest.mark.envpooltest
@pytest.mark.parametrize('env_num, batch_size', env_num_args)
class TestPoolEnvManagerV2:

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
        env_manager = PoolEnvManagerV2(env_manager_cfg)
        assert env_manager._closed
        ready_obs = env_manager.launch()
        env_id = list(ready_obs.keys())
        for count in range(5):
            action = {i: np.random.randint(4) for i in env_id}
            action_send = np.array([action[i] for i in action.keys()])
            env_id_send = np.array(list(action.keys()))
            env_manager.send_action(action_send, env_id_send)
            next_obs, rew, done, info = env_manager.receive_data()
            assert next_obs.shape == (env_manager_cfg.batch_size, 4, 84, 84)
            assert rew.shape == (env_manager_cfg.batch_size, )
            assert done.shape == (env_manager_cfg.batch_size, )
            assert info['env_id'].shape == (env_manager_cfg.batch_size, )
        env_manager.close()
        assert env_manager._closed


if __name__ == "__main__":
    TestPoolEnvManagerV2().test_naive(16, 8)
