import pytest
from dizoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
import numpy as np


@pytest.mark.envtest
class TestExpertAction:

    def test_expert_action(self):
        env = TicTacToeEnv()
        env.reset()
        print('init board state: ')
        env.render()
        # TODO(pu): How to fully test all cases
        # case 1
        env.board = np.array([[1, 2, 1], [1, 2, 0], [0, 0, 2]])
        env.current_player = 1
        assert 6 == env.expert_action()
        # case 2
        env.board = np.array([[1, 2, 1], [2, 2, 0], [1, 0, 0]])
        env.current_player = 1
        assert env.expert_action() in [5, 7]
        # case 3
        env.board = np.array([[1, 2, 1], [1, 2, 2], [0, 0, 1]])
        env.current_player = 2
        assert 7 == env.expert_action()
        # case 4
        env.board = np.array([[1, 2, 1], [1, 0, 2], [0, 0, 0]])
        env.current_player = 2
        assert 6 == env.expert_action()
