import pytest
from dizoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv


@pytest.mark.envtest
class TestTicTacToeEnv:
    def test_naive(self):
        env = TicTacToeEnv()
        env.reset()
        print('init board state: ')
        env.render()
        done = False
        while True:
            """player 1"""
            action = env.human_to_action()
            # action = env.random_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break

            """player 2"""
            action = env.expert_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

test = TestTicTacToeEnv()
test.test_naive()