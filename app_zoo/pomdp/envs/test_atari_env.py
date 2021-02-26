# import pytest
import numpy as np
from easydict import EasyDict
from app_zoo.pomdp.envs import PomdpAtariEnv
import gym


def test_env():
    cfg = {'env_id': 'Pong-ramNoFrameskip-v4',
           'frame_stack': 4,
           'is_train': True,
           'warp_frame': False,
           'clip_reward': False,
           'use_ram': True,
           'render': True,
           'pomdp': dict(
               noise_scale=0.001,
               zero_p=0.1,
               reward_noise=0.01,
               duplicate_p=0.2
           )
           }

    cfg = EasyDict(cfg)
    pong_env = PomdpAtariEnv(cfg)
    pong_env.seed(0)
    obs = pong_env.reset()
    act_dim = pong_env.info().act_space.shape[0]
    while True:
        random_action = np.random.choice(range(act_dim), size=(1, ))
        timestep = pong_env.step(random_action)
        print(timestep.obs.shape)
        assert timestep.reward.shape == (1, )
        assert isinstance(timestep, tuple)
        if timestep.done:
            assert 'final_eval_reward' in timestep.info, timestep.info
            break
    pong_env.close()


if __name__ == "__main__":
    test_env()
