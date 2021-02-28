import random
import time
from collections import namedtuple

import pytest
import torch
import numpy as np
from easydict import EasyDict
from functools import partial
from nervex.envs.env.base_env import BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.worker.actor.env_manager.subprocess_env_manager import SubprocessEnvManager, SyncSubprocessEnvManager


class EnvException(Exception):
    pass


@pytest.fixture(scope='module')
def setup_exception():
    return EnvException


class FakeEnv(object):

    def __init__(self, cfg):
        self._target_step = random.randint(4, 8) * 2
        self._current_step = 0
        self._name = cfg['name']
        self._stat = None
        self._seed = 0
        self._data_count = 0

    def reset(self, stat):
        if isinstance(stat, str) and stat == 'error':
            raise EnvException("reset error: {}".format(stat))
        self._current_step = 0
        self._stat = stat

    def step(self, action):
        if isinstance(action, str) and action == 'error':
            raise EnvException("env error, current step {}".format(self._current_step))
        if isinstance(action, str) and action == 'catched_error':
            return BaseEnvTimestep(None, None, True, {'abnormal': True})
        obs = to_ndarray(torch.randn(3))
        reward = to_ndarray(torch.randint(0, 2, size=[1]).numpy())
        done = self._current_step >= self._target_step
        simulation_time = random.uniform(0.5, 1.5)
        info = {'name': self._name, 'time': simulation_time, 'tgt': self._target_step, 'cur': self._current_step}
        time.sleep(simulation_time)
        self._current_step += simulation_time
        self._data_count += 1
        return BaseEnvTimestep(obs, reward, done, info)

    def info(self):
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T((3, ), {
                'min': [-1.0, -1.0, -8.0],
                'max': [1.0, 1.0, 8.0],
                'dtype': np.float32,
            }, None, None),
            act_space=T((1, ), {
                'min': -2.0,
                'max': 2.0,
            }, None, None),
            rew_space=T((1, ), {
                'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
                'max': -0.0,
            }, None, None),
        )

    def close(self):
        pass

    def seed(self, seed):
        self._seed = seed

    @property
    def name(self):
        return self._name

    def user_defined(self):
        pass


class FakeAsyncEnv(FakeEnv):

    def reset(self, stat):
        super().reset(stat)
        time.sleep(random.randint(2, 4))
        return to_ndarray(torch.randn(3))


class FakeModel(object):

    def forward(self, obs):
        if random.random() > 0.5:
            return {k: [] for k in obs}
        else:
            env_num = len(obs)
            exec_env = random.randint(1, env_num + 1)
            keys = list(obs.keys())[:exec_env]
            return {k: [] for k in keys}


@pytest.fixture(scope='class')
def setup_model_type():
    return FakeModel


def get_manager_cfg(shared_memory: bool):
    env_num = 4
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
            'shared_memory': shared_memory
        } for i in range(env_num)],
        'env_num': env_num,
        'episode_num': 2,
    }
    return EasyDict(manager_cfg)


def pytest_generate_tests(metafunc):
    if "setup_async_manager_cfg" in metafunc.fixturenames:
        manager_cfgs = []
        # for b in [True, False]:
        for b in [False]:
            manager_cfg = get_manager_cfg(b)
            manager_cfg['env_fn'] = FakeAsyncEnv
            manager_cfgs.append(manager_cfg)
        metafunc.parametrize("setup_async_manager_cfg", manager_cfgs)


@pytest.fixture(scope='class')
def setup_sync_manager_cfg():
    manager_cfg = get_manager_cfg(False)
    manager_cfg['env_fn'] = FakeEnv
    return manager_cfg
