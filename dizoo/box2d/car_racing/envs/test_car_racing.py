import pytest
import numpy as np
import gym
from easydict import EasyDict
from dizoo.box2d.car_racing.envs import CarRacingEnv


@pytest.mark.envtest
class TestAtariEnv:

    def test_car_racing(self):
        cfg = {'env_id': 'CarRacing-v0', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        env = CarRacingEnv(cfg)
        env.seed(0)
        obs = env.reset()
        assert obs.shape == (cfg.frame_stack, 84, 84)

        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        for i in range(10):
            random_action = env.action_space.sample()
            print(random_action)
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.reward.shape == (1,)
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
            # assert isinstance(timestep, tuple)
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
