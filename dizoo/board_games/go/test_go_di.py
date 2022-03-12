import pytest
from easydict import EasyDict
import numpy as np
from dizoo.board_games.go.go_env_di import GoDIEnv


@pytest.mark.envtest
class TestGoDIEnv:

    def test_naive(self):
        env = GoDIEnv(board_size=9, komi=7.5)
        obs, reward, done, info = env.reset()
        # env.render()
        # while True:
        for i in range(100):
            """player 0"""
            # action = env.human_to_action()
            action = env.random_action()
            obs, reward, done, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, float)
            # env.render()
            if done:
                # env.render()
                if reward > 0:
                    print('human player win')
                else:
                    print('draw')
                break

            """player 1"""
            action = env.random_action()
            # print('computer player (player1) take action: ' + f'{action}')
            obs, reward, done, info = env.step(action)
            # print(f'After the computer player (player1) took action: {action}, the current board state is:')
            # env.render()
            if done:
                if reward > 0:
                    print('computer player win')
                else:
                    print('draw')
                break

