import pytest
from easydict import EasyDict
import numpy as np
from dizoo.box2d.bipedalwalker.envs import BipedalWalkerEnv


@pytest.mark.envtest
class TestBipedalWalkerEnv:

    def test_naive(self):
        env = BipedalWalkerEnv(EasyDict({'act_scale': True, 'rew_clip': True, 'replay_path': None}))
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        assert obs.shape == (24, )
        for i in range(10):
            random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (24, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
            # assert isinstance(timestep, tuple)
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
