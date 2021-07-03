import random
import time
from collections import namedtuple

import pytest
import torch
import numpy as np
from easydict import EasyDict
from functools import partial
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.env.base_env import BaseEnvTimestep, BaseEnvInfo
from ding.envs.env_manager.base_env_manager import EnvState
from ding.envs.env_manager import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import WatchDog, deep_merge_dicts


class EnvException(Exception):
    pass


@pytest.fixture(scope='module')
def setup_exception():
    return EnvException


@pytest.fixture(scope='module')
def setup_watchdog():
    return WatchDog


class FakeEnv(object):

    def __init__(self, cfg):
        self._target_time = random.randint(3, 6)
        self._current_time = 0
        self._name = cfg['name']
        self._stat = None
        self._seed = 0
        self._data_count = 0
        self.timeout_flag = False
        self._launched = False
        self._state = EnvState.INIT

    def reset(self, stat):
        if isinstance(stat, str) and stat == 'error':
            self.dead()
        if isinstance(stat, str) and stat == "timeout":
            if self.timeout_flag:  # after step(), the reset can hall with status of timeout
                time.sleep(5)
        if isinstance(stat, str) and stat == "block":
            self.block()

        self._launched = True
        self._current_time = 0
        self._stat = stat
        self._state = EnvState.RUN

    def step(self, action):
        assert self._launched
        assert not self._state == EnvState.ERROR
        self.timeout_flag = True  # after one step, enable timeout flag
        if isinstance(action, str) and action == 'error':
            self.dead()
        if isinstance(action, str) and action == 'catched_error':
            return BaseEnvTimestep(None, None, True, {'abnormal': True})
        if isinstance(action, str) and action == "timeout":
            if self.timeout_flag:  # after step(), the reset can hall with status of timeout
                time.sleep(3)
        if isinstance(action, str) and action == 'block':
            self.block()
        obs = to_ndarray(torch.randn(3))
        reward = to_ndarray(torch.randint(0, 2, size=[1]).numpy())
        done = self._current_time >= self._target_time
        if done:
            self._state = EnvState.DONE
        simulation_time = random.uniform(0.5, 1)
        info = {'name': self._name, 'time': simulation_time, 'tgt': self._target_time, 'cur': self._current_time}
        time.sleep(simulation_time)
        self._current_time += simulation_time
        self._data_count += 1
        return BaseEnvTimestep(obs, reward, done, info)

    def dead(self):
        self._state = EnvState.ERROR
        raise EnvException("env error, current time {}".format(self._current_time))

    def block(self):
        self._state = EnvState.ERROR
        time.sleep(1000)

    def info(self):
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T((3, ), {
                'min': [-1.0, -1.0, -8.0],
                'max': [1.0, 1.0, 8.0],
                'dtype': np.float32,
            }, None),
            act_space=T((1, ), {
                'min': -2.0,
                'max': 2.0,
            }, None),
            rew_space=T((1, ), {
                'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
                'max': -0.0,
            }, None),
        )

    def close(self):
        self._launched = False
        self._state = EnvState.INIT

    def seed(self, seed):
        self._seed = seed

    @property
    def name(self):
        return self._name

    def user_defined(self):
        pass

    def __repr__(self):
        return self._name


class FakeAsyncEnv(FakeEnv):

    def reset(self, stat):
        super().reset(stat)
        time.sleep(random.randint(1, 3))
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


def get_manager_cfg(env_num=4):
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
        } for i in range(env_num)],
        'episode_num': 2,
        'reset_timeout': 10,
        'step_timeout': 8,
        'max_retry': 5,
    }
    return EasyDict(manager_cfg)


@pytest.fixture(scope='function')
def setup_base_manager_cfg():
    manager_cfg = get_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeEnv, cfg=c) for c in env_cfg]
    return deep_merge_dicts(BaseEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_sync_manager_cfg():
    manager_cfg = get_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeEnv, cfg=c) for c in env_cfg]
    return deep_merge_dicts(SyncSubprocessEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_async_manager_cfg():
    manager_cfg = get_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeAsyncEnv, cfg=c) for c in env_cfg]
    manager_cfg['shared_memory'] = False
    manager_cfg['connect_timeout'] = 30
    return deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), EasyDict(manager_cfg))
