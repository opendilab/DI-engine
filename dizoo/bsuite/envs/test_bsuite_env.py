from time import time
import pytest
import numpy as np
from easydict import EasyDict
from dizoo.bsuite.envs import BSuiteEnv


@pytest.mark.envtest
class TestBSuiteEnv:

    def test_memory_len(self):
        cfg = {'env': {'env_id': 'memory_len/0'}}
        cfg = EasyDict(cfg)
        memory_len_env = BSuiteEnv(cfg)
        memory_len_env.seed(0)
        obs = memory_len_env.reset()
        assert obs.shape == (3, )
        print(memory_len_env.info())
        act_dim = memory_len_env.info().act_space.shape[0]
        while True:
            random_action = np.random.choice(range(act_dim), size=(1, ))
            timestep = memory_len_env.step(random_action)
            assert timestep.obs.shape == (3, )
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(memory_len_env.info(), 'final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        memory_len_env.close()

    def test_cartpole_swingup(self):
        cfg = {'env': {'env_id': 'cartpole_swingup/0'}}
        cfg = EasyDict(cfg)
        bandit_noise_env = BSuiteEnv(cfg)
        bandit_noise_env.seed(0)
        obs = bandit_noise_env.reset()
        assert obs.shape == (8, )
        print(bandit_noise_env.info())
        act_dim = bandit_noise_env.info().act_space.shape[0]
        while True:
            random_action = np.random.choice(range(act_dim), size=(1, ))
            timestep = bandit_noise_env.step(random_action)
            assert timestep.obs.shape == (8, )
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(bandit_noise_env.info(), 'final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        bandit_noise_env.close()

    def test_info(self):
        cfg = {'env': {'env_id': 'memory_len/0'}}
        cfg = EasyDict(cfg)
        memory_len_env = BSuiteEnv(cfg)
        info_dict = memory_len_env.info()
        print(info_dict)
        memory_len_env.close()
