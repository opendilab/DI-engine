import pytest
from dizoo.board_games.atari.atari_env_di import AtariDIEnv
from easydict import EasyDict

cfg = EasyDict(env_id='PongNoFrameskip-v4',
               frame_stack=4,
               is_train=True)

@pytest.mark.envtest
class TestChessDIEnv:

    def test_naive(self):

        env = AtariDIEnv(cfg)
        obs, reward, done, info = env.reset()
        # env.render()
        while True:
            # action = env.human_to_action()
            action = env.random_action()

            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                print(info)
                break

            action = env.random_action()
            # print('computer player ' + env.action_to_string(action))
            print('computer player (player1) take action: ' + f'{action}')

            obs, reward, done, info = env.step(action)
            print(f'After the computer player (player1) took action: {action}, the current board state is:')
            # env.render()
            if done:
                print(info)
                break
