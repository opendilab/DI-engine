import pytest
from dizoo.atari.envs.atari_muzero_env import AtariMuZeroEnv
from easydict import EasyDict

cfg = EasyDict(env_name='PongNoFrameskip-v4',
               frame_skip=4,
               frame_stack=4,
               episode_life=True,
               obs_shape=(12, 96, 96),
               gray_scale=False,
               discount=0.997,
               cvt_string=True,
               max_episode_steps=1.08e5,
               game_wrapper=True,
               )


@pytest.mark.envtest
class TestAtariMuZeroEnv:
    def test_naive(self):
        env = AtariMuZeroEnv(cfg)
        env.reset()
        # env.render()
        while True:
            action = env.random_action()
            # action = env.human_to_action()
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                print(info)
                break
