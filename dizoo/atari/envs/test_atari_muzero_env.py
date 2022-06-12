import pytest
from dizoo.atari.envs.atari_muzero_env import AtariMuZeroEnv
from easydict import EasyDict

cfg = EasyDict(env_name='PongNoFrameskip-v4',
               frame_skip=4,
               frame_stack=4,
               max_moves=1e6,
               episode_life=True,
               obs_shape=(12, 96, 96),
               gray_scale=False,
               discount=0.997,
               cvt_string=True,
               is_train=True)


@pytest.mark.envtest
class TestAtariMuZeroEnv:
    def test_naive(self):
        env = AtariMuZeroEnv(cfg)
        obs, reward, done, info = env.reset()
        # env.render()
        print('=' * 20)
        print('In atari, player 1 = player 2')
        print('=' * 20)
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                print(info)
                print('=' * 20)
                print('In atari, player 1 = player 2')
                print('=' * 20)
                break

            action = env.random_action()
            # action = env.human_to_action()
            print('player 2: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                print(info)
                print('=' * 20)
                print('In atari, player 1 = player 2')
                print('=' * 20)
                break
