import time
import signal
import pytest
import torch
import numpy as np

from ding.envs.env_manager.base_env_manager import BaseEnvManager, EnvState
from ding.envs.env_manager.gym_vector_env_manager import GymVectorEnvManager
from gym.vector.async_vector_env import AsyncState


@pytest.mark.unittest
class TestGymVectorEnvManager:

    def test_naive(self, setup_gym_vector_manager_cfg):
        env_fn = setup_gym_vector_manager_cfg.pop('env_fn')
        env_manager = GymVectorEnvManager(env_fn, setup_gym_vector_manager_cfg)
        env_manager.seed([314 for _ in range(env_manager.env_num)])
        # Test reset
        obs = env_manager.reset()
        assert not env_manager._closed
        assert env_manager._env_manager._state == AsyncState.DEFAULT
        # Test arribute
        with pytest.raises(AttributeError):
            _ = env_manager.xxx
        with pytest.raises(RuntimeError):
            env_manager.user_defined()
        # Test step
        count = 1
        start_time = time.time()
        while not env_manager.done:
            env_id = env_manager.ready_obs.keys()
            assert all(env_manager._env_episode_count[i] < env_manager._episode_num for i in env_id)
            action = {i: np.random.randn(4) for i in env_id}
            timestep = env_manager.step(action)
            assert len(timestep) == len(env_id)
            print('Count {}'.format(count))
            print([v.info for v in timestep.values()])
            print([v.done for v in timestep.values()])
            count += 1
        end_time = time.time()
        print('total step time: {}'.format(end_time - start_time))
        assert all(env_manager._env_episode_count[i] == env_manager._episode_num for i in env_id)

        # Test close
        assert not env_manager._closed
        env_manager.close()
        assert env_manager._closed
        assert env_manager._env_ref._state == EnvState.INIT
        # assert all([not env_manager._envs[env_id]._launched for env_id in range(env_manager.env_num)])
        # assert all([env_manager._env_states[env_id] == EnvState.VOID for env_id in range(env_manager.env_num)])
        with pytest.raises(AssertionError):
            env_manager.reset([])
        with pytest.raises(AssertionError):
            env_manager.step([])
