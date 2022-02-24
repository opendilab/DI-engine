import pytest
import numpy as np
from easydict import EasyDict
from dizoo.classic_control.pendulum.envs import PendulumEnv


@pytest.mark.envtest
class TestPendulumEnv:

    def test_naive(self):
        env = PendulumEnv(EasyDict({'act_scale': True}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3, )
        for i in range(10):
            random_action = np.tanh(np.random.random(1))
            timestep = env.step(random_action)
            assert timestep.obs.shape == (3, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
            # assert isinstance(timestep, tuple)
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
