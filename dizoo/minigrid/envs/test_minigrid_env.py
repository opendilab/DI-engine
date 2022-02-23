import pytest
import os
import numpy as np
from dizoo.minigrid.envs import MiniGridEnv
from easydict import EasyDict
import copy

# The following two cfg can be tested through TestMiniGridAKTDTnv
config = dict(
    env_id='MiniGrid-AKTDT-13x13-v0',
    flat_obs=True,
)
cfg = EasyDict(copy.deepcopy(config))
cfg.cfg_type = 'MiniGridEnvDict'

config2 = dict(
    env_id='MiniGrid-AKTDT-7x7-1-v0',
    flat_obs=True,
)
cfg2 = EasyDict(copy.deepcopy(config2))
cfg2.cfg_type = 'MiniGridEnvDict'


@pytest.mark.envtest
class TestMiniGridEnv:

    def test_naive(self):
        env = MiniGridEnv(MiniGridEnv.default_config())
        env.seed(314)
        path = './video'
        if not os.path.exists(path):
            os.mkdir(path)
        env.enable_save_replay(path)
        assert env._seed == 314
        obs = env.reset()
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(env._max_step):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2739, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            if timestep.done:
                env.reset()
        print(env.info())
        env.close()


@pytest.mark.envtest
class TestMiniGridAKTDTnv:

    def test_adtkt_13(self):
        env = MiniGridEnv(cfg2)
        env.seed(314)
        path = './video'
        if not os.path.exists(path):
            os.mkdir(path)
        env.enable_save_replay(path)
        assert env._seed == 314
        obs = env.reset()
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(env._max_step):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2667, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            if timestep.done:
                env.reset()
        print(env.info())
        env.close()

    def test_adtkt_7(self):
        env = MiniGridEnv(cfg2)
        env.seed(314)
        path = './video'
        if not os.path.exists(path):
            os.mkdir(path)
        env.enable_save_replay(path)
        assert env._seed == 314
        obs = env.reset()
        act_val = env.info().act_space.value
        min_val, max_val = act_val['min'], act_val['max']
        for i in range(env._max_step):
            random_action = np.random.randint(min_val, max_val, size=(1, ))
            timestep = env.step(random_action)
            print(timestep)
            print(timestep.obs.max())
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2619, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.info().rew_space.value['min']
            assert timestep.reward <= env.info().rew_space.value['max']
            if timestep.done:
                env.reset()
        print(env.info())
        env.close()
