import pytest
from dizoo.board_games.chess.envs.chess_env_game import ChessEnv


@pytest.mark.envtest
class TestChessEnv:

    def test_naive(self):
        env = ChessEnv()
        env.reset()
        print('init board state: ')
        env.render()
        for i in range(100):
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()
            print('player 1: ', action)
            obs, reward, done, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, int)
            # env.render()
            if done:
                if done:
                    if reward > 0:
                        print('player 1 (human player) win')
                    else:
                        print('draw')
                    break

            """player 2"""
            action = env.random_action()
            print('player 2 (computer player): ', action)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

