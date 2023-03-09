import easydict
import numpy
import pytest

from ding.utils import get_vi_sequence
from dizoo.maze.envs.maze_env import Maze


@pytest.mark.unittest
class TestBFSHelper:

    def test_bfs(self):

        def load_env(seed):
            ccc = easydict.EasyDict({'size': 16})
            e = Maze(ccc)
            e.seed(seed)
            e.reset()
            return e

        env = load_env(314)
        start_obs = env.process_states(env._get_obs(), env.get_maze_map())
        vi_sequence, track_back = get_vi_sequence(env, start_obs)
        assert vi_sequence.shape[1:] == (16, 16)
        assert track_back[0][0].shape == (16, 16, 3)
        assert isinstance(track_back[0][1], numpy.int32)
