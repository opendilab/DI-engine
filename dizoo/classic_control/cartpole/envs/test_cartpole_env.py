import pytest
import numpy as np
from dizoo.classic_control.cartpole.envs import CartPoleEnv


@pytest.mark.unittest
class TestCartPoleEnv:

    def test_naive(self):
        env = CartPoleEnv({})
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (4, )
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for _ in range(5):
            env.reset()
            np.random.seed(314)
            print('=' * 60)
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
        print(env.info())
        env.close()
