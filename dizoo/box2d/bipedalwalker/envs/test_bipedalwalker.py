import pytest
from easydict import EasyDict
import numpy as np
from dizoo.box2d.bipedalwalker.envs import BipedalWalkerEnv


@pytest.mark.unittest
class TestBipedalWalkerEnv:

    def test_naive(self):
        env = BipedalWalkerEnv(EasyDict({'act_scale': True, 'rew_clip': True, 'replay_path': None}))
        env.seed(123)
        assert env._seed == 123
        obs = env.reset()
        assert obs.shape == (24, )
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(10):
            random_action = np.random.randint(min_val, max_val, size=(4, ))
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (24, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            # assert isinstance(timestep, tuple)
        print(env.info())
        env.close()
