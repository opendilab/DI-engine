import numpy as np
import pytest
from dizoo.frozen_lake.envs import FrozenLakeEnv
from easydict import EasyDict


@pytest.mark.envtest
class TestGymHybridEnv:

    def test_my_lake(self):
        env = FrozenLakeEnv(
            EasyDict({
                'env_id': 'FrozenLake-v1',
                'desc': None,
                'map_name': "4x4",
                'is_slippery': False,
            })
        )
        for _ in range(5):
            env.seed(314, dynamic_seed=False)
            assert env._seed == 314
            obs = env.reset()
            assert obs.shape == (
                16,
            ), "Considering the one-hot encoding format, your observation should have a dimensionality of 16."
            for i in range(10):
                env.enable_save_replay("./video")
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.array([env.action_space.sample()])
                else:
                    random_action = env.random_action()
                timestep = env.step(random_action)
                print(timestep)
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (16, )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high

        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
