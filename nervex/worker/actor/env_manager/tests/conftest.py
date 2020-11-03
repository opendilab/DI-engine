import random
import time
from collections import namedtuple

import pytest
import torch
from easydict import EasyDict


class EnvException(Exception):
    pass


@pytest.fixture(scope='module')
def setup_exception():
    return EnvException


class FakeEnv(object):
    timestep = namedtuple('timestep', ['obs', 'rew', 'done', 'info'])

    def __init__(self, cfg):
        self._target_step = random.randint(4, 8) * 3
        self._current_step = 0
        self._name = cfg['name']
        self._stat = None
        self._seed = 0
        self._data_count = 0

    def reset(self, stat):
        self._current_step = 0
        self._stat = stat

    def step(self, action):
        if isinstance(action, str) and action == 'error':
            raise EnvException("env error")
        obs = torch.randn(3)
        reward = torch.randint(0, 2, size=[1])
        done = self._current_step >= self._target_step
        simulation_time = random.uniform(0.5, 1.5)
        info = {'name': self._name, 'time': simulation_time, 'tgt': self._target_step, 'cur': self._current_step}
        time.sleep(simulation_time)
        self._current_step += simulation_time
        self._data_count += 1
        return FakeEnv.timestep(obs, reward, done, info)

    def close(self):
        pass

    def seed(self, seed):
        self._seed = seed

    def info(self):
        return {'name': 'FakeEnv'}

    @property
    def name(self):
        return self._name


class FakeAsyncEnv(FakeEnv):
    timestep = namedtuple('timestep', ['obs', 'rew', 'done', 'info'])

    def reset(self, stat):
        super().reset(stat)
        time.sleep(random.randint(5, 8))


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


def get_manager_cfg():
    env_num = 4
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
        } for i in range(env_num)],
        'env_num': env_num,
        'episode_num': 2,
    }
    return EasyDict(manager_cfg)


@pytest.fixture(scope='class')
def setup_async_manager_cfg():
    manager_cfg = get_manager_cfg()
    manager_cfg['env_fn'] = FakeAsyncEnv
    return manager_cfg


@pytest.fixture(scope='class')
def setup_sync_manager_cfg():
    manager_cfg = get_manager_cfg()
    manager_cfg['env_fn'] = FakeEnv
    return manager_cfg
