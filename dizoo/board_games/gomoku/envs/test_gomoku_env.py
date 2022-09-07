import pytest
from dizoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


@pytest.mark.envtest
class TestGomokuEnv:

    def test_naive(self):
        env = GomokuEnv()
        obs = env.reset()
        print('init board state: ')
        env.render()
        done = False
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break

            action = env.random_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break
