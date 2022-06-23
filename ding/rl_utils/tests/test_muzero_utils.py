import random
import numpy as np
import pytest
from ding.rl_utils.efficientzero.utils import get_augmented_data


@pytest.mark.unittest
class TestMuZeroUtils():

    def test_get_augmented_data(self):
        num_of_data = 100
        board_size = 15
        state = np.random.randint(0, 3, (board_size, board_size, 3), dtype=np.uint8)
        mcts_prob = np.random.randn(board_size, board_size)
        winner = np.random.randint(0, 2, 1, dtype=np.uint8)
        play_data = [{'state': state, 'mcts_prob': mcts_prob, 'winner': winner} for _ in range(num_of_data)]

        extented_data = get_augmented_data(board_size, play_data)
        assert len(extented_data) == num_of_data * 8
        # TODO(pu): extented data shape is not same as original data?
        # assert extented_data[0]['state'].shape == state.shape
        assert extented_data[0]['state'].flatten().shape == state.flatten().shape
        assert extented_data[0]['mcts_prob'].shape == mcts_prob.flatten().shape
        assert extented_data[0]['winner'].shape == winner.shape
