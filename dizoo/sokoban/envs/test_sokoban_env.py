from easydict import EasyDict
import pytest
import numpy as np
from dizoo.sokoban.envs.sokoban_env import SokobanEnv


@pytest.mark.envtest
class TestSokoban:

    def test_sokoban(self):
        env = SokobanEnv(EasyDict({'env_id': 'Sokoban-v0'}))
        env.reset()
        for i in range(100):
            action = np.random.randint(8)
            timestep = env.step(np.array(action))
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (160, 160, 3)
            print(timestep.info)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                env.reset()
        env.close()
