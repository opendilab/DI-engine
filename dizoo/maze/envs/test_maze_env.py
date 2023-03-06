import pytest
import os
import numpy as np
from dizoo.maze.envs.maze_env import Maze
from easydict import EasyDict
import copy


@pytest.mark.envtest
class TestMazeEnv:

    def test_maze(self):
        env = Maze(EasyDict({'size': 16}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (16, 16, 3)
        min_val, max_val = 0, 3
        for i in range(100):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            if timestep.done:
                env.reset()
        env.close()
