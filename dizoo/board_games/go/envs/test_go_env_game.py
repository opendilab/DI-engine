import pytest
from dizoo.board_games.go.envs.go_env_game import GoEnv


@pytest.mark.envtest
class TestGoEnv:

    def test_naive(self):
        env = GoEnv(board_size=9, komi=7.5)
        print('NOTE：actions are counted by column, such as action 9, which is the second column and the first row')
        env.reset()
        # env.render()
        for i in range(100):
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()
            print('player 1 (black_0): ', action)
            obs, reward, done, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, float)
            # env.render()
            if done:
                if reward > 0:
                    print('player 1 (black_0) win')
                else:
                    print('draw')
                break

            """player 2"""
            action = env.random_action()
            print('player 2 (white_0): ', action)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                if reward > 0:
                    print('player 2 (white_0) win')
                else:
                    print('draw')
                break
