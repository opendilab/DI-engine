import pytest
import torch
from .pendulum_env import PendulumEnv


@pytest.mark.unittest
class TestPendulumEnv:

    def test_naive(self):
        env = PendulumEnv({})
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3, )
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(10):
            random_action = torch.rand(1) * (max_val - min_val) + min_val
            timestep = env.step(random_action)
            assert timestep.obs.shape == (3, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            assert isinstance(timestep, tuple)
        print(env.info())
        env.close()
