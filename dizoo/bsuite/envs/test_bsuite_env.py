from time import time
import pytest
import numpy as np
from easydict import EasyDict
from dizoo.bsuite.envs import BSuiteEnv


@pytest.mark.unittest
class TestBSuiteEnv:

    def test_memory_len(self):
        cfg = {'env_id': 'memory_len/0'}
        cfg = EasyDict(cfg)
        memory_len_env = BSuiteEnv(cfg)
        memory_len_env.seed(0)
        obs = memory_len_env.reset()
        assert obs.shape == (1, 3)
        print(memory_len_env.info())
        act_dim = memory_len_env.info().act_space.shape[0]
        while True:
            random_action = np.random.choice(range(act_dim), size=(1, ))
            timestep = memory_len_env.step(random_action)
            assert timestep.obs.shape == (1, 3)
            assert timestep.reward.shape == (1, )
            if timestep.done:
                assert 'final_eval_reward' in timestep.info, timestep.info
                break
        print(memory_len_env.info(), 'final_eval_reward: {}'.format(timestep.info['final_eval_reward']))
        memory_len_env.close()

    def test_info(self):
        cfg = {'env_id': 'memory_len/0'}
        cfg = EasyDict(cfg)
        memory_len_env = BSuiteEnv(cfg)
        info_dict = memory_len_env.info()
        print(info_dict)
        memory_len_env.close()
