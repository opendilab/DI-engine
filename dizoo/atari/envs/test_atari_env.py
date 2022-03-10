import pytest
import numpy as np
import gym
from easydict import EasyDict
import atari_py

from dizoo.atari.envs import AtariEnv, AtariEnvMR


@pytest.mark.envtest
class TestAtariEnv:

    def test_pong(self):
        cfg = {'env_id': 'PongNoFrameskip-v4', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        pong_env = AtariEnv(cfg)
        pong_env.seed(0)
        obs = pong_env.reset()
        assert obs.shape == (cfg.frame_stack, 84, 84)
        act_dim = pong_env.action_space.n
        i = 0
        while True:
            # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
            # can generate legal random action.
            if i < 10:
                random_action = np.random.choice(range(act_dim), size=(1, ))
                i += 1
            else:
                random_action = pong_env.random_action()
            timestep = pong_env.step(random_action)
            assert timestep.obs.shape == (cfg.frame_stack, 84, 84)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(pong_env.observation_space, pong_env.action_space, pong_env.reward_space)
        print('final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        pong_env.close()

    def test_montezuma_revenge(self):
        cfg = {'env_id': 'MontezumaRevengeDeterministic-v4', 'frame_stack': 4, 'is_train': True}
        cfg = EasyDict(cfg)
        mr_env = AtariEnvMR(cfg)
        mr_env.seed(0)
        obs = mr_env.reset()
        assert obs.shape == (cfg.frame_stack, 84, 84)
        act_dim = mr_env.action_space.n
        i = 0
        while True:
            if i < 10:
                random_action = np.random.choice(range(act_dim), size=(1, ))
                i += 1
            else:
                random_action = mr_env.random_action()
            timestep = mr_env.step(random_action)
            assert timestep.obs.shape == (cfg.frame_stack, 84, 84)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(mr_env.observation_space, mr_env.action_space, mr_env.reward_space)
        print('final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        mr_env.close()
