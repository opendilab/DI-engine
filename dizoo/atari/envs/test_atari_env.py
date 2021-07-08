import pytest
import numpy as np
import gym
from easydict import EasyDict
from dizoo.atari.envs import AtariEnv, AtariEnvMR


@pytest.mark.unittest
class TestAtariEnv:

    def test_pong(self):
        cfg = {'env_id': 'PongNoFrameskip-v4', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        pong_env = AtariEnv(cfg)
        pong_env.seed(0)
        obs = pong_env.reset()
        assert obs.shape == (cfg.frame_stack, 84, 84)
        act_dim = pong_env.info().act_space.shape[0]
        while True:
            random_action = np.random.choice(range(act_dim), size=(1, ))
            timestep = pong_env.step(random_action)
            assert timestep.obs.shape == (cfg.frame_stack, 84, 84)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(pong_env.info(), 'final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        pong_env.close()

    def test_montezuma_revenge(self):
        cfg = {'env_id': 'MontezumaRevengeDeterministic-v4', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        mr_env = AtariEnvMR(cfg)
        mr_env.seed(0)
        obs = mr_env.reset()
        assert obs.shape == (cfg.frame_stack, 84, 84)
        act_dim = mr_env.info().act_space.shape[0]
        while True:
            random_action = np.random.choice(range(act_dim), size=(1, ))
            timestep = mr_env.step(random_action)
            assert timestep.obs.shape == (cfg.frame_stack, 84, 84)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(mr_env.info(), 'final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        mr_env.close()

    def test_info(self):
        cfg = {'env_id': 'PongNoFrameskip-v4', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        pong_env = AtariEnv(cfg)
        info_dict = pong_env.info()
        print(info_dict)
        pong_env.close()
