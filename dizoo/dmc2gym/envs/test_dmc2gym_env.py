import pytest
import numpy as np
from easydict import EasyDict
from dizoo.dmc2gym.envs import DMC2GymEnv
from torch import float32


@pytest.mark.envtest
class TestDMC2GymEnv:

    def test_naive(self):
        env = DMC2GymEnv(EasyDict({
            "domain_name": "cartpole",
            "task_name": "balance",
            "frame_skip": 2,
        }))
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (
            3,
            100,
            100,
        )
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
            for i in range(10):
                # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
                # can generate legal random action.
                if i < 5:
                    random_action = np.array(env.action_space.sample(), dtype=np.float32)
                else:
                    random_action = env.random_action()
                timestep = env.step(random_action)
                print(timestep)
                assert isinstance(timestep.obs, np.ndarray)
                assert isinstance(timestep.done, bool)
                assert timestep.obs.shape == (
                    3,
                    100,
                    100,
                )
                assert timestep.reward.shape == (1, )
                assert timestep.reward >= env.reward_space.low
                assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
