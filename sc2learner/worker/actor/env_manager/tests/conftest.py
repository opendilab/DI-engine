import pytest
import time
import random
import torch
from easydict import EasyDict
from collections import namedtuple


class FakeEnv(object):
    timestep = namedtuple('timestep', ['obs', 'rew', 'done', 'info'])

    def __init__(self, cfg):
        self._target_step = random.randint(4, 8) * 3
        self._current_step = 0
        self._name = cfg['name']

    def reset(self, stat):
        self._current_step = 0
        self._stat = stat

    def step(self, action):
        obs = torch.randn(3)
        reward = torch.randint(0, 2, size=[1])
        done = self._current_step >= self._target_step
        simulation_time = random.uniform(0.5, 1.5)
        info = {'name': self._name, 'time': simulation_time, 'tgt': self._target_step, 'cur': self._current_step}
        time.sleep(simulation_time)
        self._current_step += simulation_time
        return FakeEnv.timestep(obs, reward, done, info)

    def close(self):
        pass

    def seed(self, seed):
        self._seed = seed


# TODO(nyz) pickle can't find conftest.timestep
timestep = FakeEnv.timestep


@pytest.fixture(scope='class')
def setup_env_type():
    return FakeEnv


@pytest.fixture(scope='class')
def setup_manager_cfg(setup_env_type):
    env_num = 4
    manager_cfg = {
        'env': [{
            'name': 'name{}'.format(i),
            'type': setup_env_type
        } for i in range(env_num)],
        'env_num': env_num
    }
    return EasyDict(manager_cfg)
