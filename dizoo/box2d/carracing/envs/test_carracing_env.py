import pytest
import numpy as np
from easydict import EasyDict
from carracing_env import CarRacingEnv


@pytest.mark.envtest
@pytest.mark.parametrize('cfg', [EasyDict({'env_id': 'CarRacing-v2', 'continuous': False, 'act_scale': False})])
class TestCarRacing:

    def test_naive(self, cfg):
        env = CarRacingEnv(cfg)
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3, 96, 96)
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (3, 96, 96)
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
