import pytest
import numpy as np
from app_zoo.classic_control.cartpole.envs import CartPoleEnv


@pytest.mark.unittest
class TestCartPoleEnv:

    def test_naive(self):
        env = CartPoleEnv({})
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (4, )
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(10):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (4, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            assert isinstance(timestep, tuple)
        print(env.info())
        env.close()
