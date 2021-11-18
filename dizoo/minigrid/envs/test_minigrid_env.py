import pytest
import os
import numpy as np
from dizoo.minigrid.envs import MiniGridEnv


@pytest.mark.envtest
class TestMiniGridEnv:

    def test_naive(self):
        env = MiniGridEnv(MiniGridEnv.default_config())
        env.seed(314)
        path = './video'
        if not os.path.exists(path):
            os.mkdir(path)
        env.enable_save_replay(path)
        assert env._seed == 314
        obs = env.reset()
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(env._max_step):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2739, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            if timestep.done:
                env.reset()
        print(env.info())
        env.close()
