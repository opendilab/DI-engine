import pytest
import numpy as np
from easydict import EasyDict
from dizoo.procgen.envs import ProcgenEnv


@pytest.mark.envtest
class TestProcgenEnv:

    def test_naive(self):
        env = ProcgenEnv(EasyDict({}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (3, 64, 64)
        for i in range(10):
            random_action = np.tanh(np.random.random(1))
            timestep = env.step(random_action)
            assert timestep.obs.shape == (3, 64, 64)
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            # assert isinstance(timestep, tuple)
        print(env.info())
        env.close()
