import numpy as np
import pytest
from dizoo.frozen_lake.envs import FrozenLakeEnv
from easydict import EasyDict

@pytest.mark.envtest
class TestGymHybridEnv:
    def test_my_lake(self):
        env = FrozenLakeEnv(
            EasyDict(
                    {
                        'env_id': 'FrozenLake-v1',
                        'desc': None,
                        'map_name': "4x4",
                        'is_slippery': True,
                        'save_replay_gif': False,
                        'replay_path_gif': None,
                        'replay_path': None,
                    }
                ))
        env.seed(314,dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            if timestep.done:
                print('reset env')