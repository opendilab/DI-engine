import random
import time
from collections import namedtuple
import pytest
import torch
import numpy as np
from easydict import EasyDict
from functools import partial
import gym

from ding.envs.env.base_env import BaseEnvTimestep
from ding.envs.env_manager.base_env_manager import EnvState
from ding.envs.env_manager import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import deep_merge_dicts


class FakeEnv(object):

    def __init__(self, cfg):
        self._scale = cfg.scale
        self._target_time = random.randint(3, 6) * self._scale
        self._current_time = 0
        self._name = cfg['name']
        self._id = time.time()
        self._stat = None
        self._seed = 0
        self._data_count = 0
        self.timeout_flag = False
        self._launched = False
        self._state = EnvState.INIT
        self._dead_once = False
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -8.0]), high=np.array([1.0, 1.0, 8.0]), shape=(3, ), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1, ), dtype=np.float32)
        self.reward_space = gym.spaces.Box(
            low=-1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2), high=0.0, shape=(1, ), dtype=np.float32
        )

    def reset(self, stat=None):
        if isinstance(stat, str) and stat == 'error':
            self.dead()
        if isinstance(stat, str) and stat == 'error_once':
            # Die on every two reset with error_once stat.
            if self._dead_once:
                self._dead_once = False
                self.dead()
            else:
                self._dead_once = True
        if isinstance(stat, str) and stat == "wait":
            if self.timeout_flag:  # after step(), the reset can hall with status of timeout
                time.sleep(5)
        if isinstance(stat, str) and stat == "block":
            self.block()

        self._launched = True
        self._current_time = 0
        self._stat = stat
        self._state = EnvState.RUN
        return to_ndarray(torch.randn(3))

    def step(self, action):
        assert self._launched
        assert not self._state == EnvState.ERROR
        self.timeout_flag = True  # after one step, enable timeout flag
        if isinstance(action, str) and action == 'error':
            self.dead()
        if isinstance(action, str) and action == 'catched_error':
            return BaseEnvTimestep(None, None, True, {'abnormal': True})
        if isinstance(action, str) and action == "wait":
            if self.timeout_flag:  # after step(), the reset can hall with status of timeout
                time.sleep(3)
        if isinstance(action, str) and action == 'block':
            self.block()
        obs = to_ndarray(torch.randn(3))
        reward = to_ndarray(torch.randint(0, 2, size=[1]).numpy())
        done = self._current_time >= self._target_time
        if done:
            self._state = EnvState.DONE
        simulation_time = random.uniform(0.5, 1) * self._scale
        info = {'name': self._name, 'time': simulation_time, 'tgt': self._target_time, 'cur': self._current_time}
        time.sleep(simulation_time)
        self._current_time += simulation_time
        self._data_count += 1
        return BaseEnvTimestep(obs, reward, done, info)

    def dead(self):
        self._state = EnvState.ERROR
        raise RuntimeError("env error, current time {}".format(self._current_time))

    def block(self):
        self._state = EnvState.ERROR
        time.sleep(1000)

    def close(self):
        self._launched = False
        self._state = EnvState.INIT

    def seed(self, seed):
        self._seed = seed

    @property
    def name(self):
        return self._name

    @property
    def time_id(self):
        return self._id

    def user_defined(self):
        pass

    def __repr__(self):
        return self._name


class FakeAsyncEnv(FakeEnv):

    def reset(self, stat=None):
        super().reset(stat)
        time.sleep(random.randint(1, 3) * self._scale)
        return to_ndarray(torch.randn(3))


class FakeGymEnv(FakeEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = "fake metadata"
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(4, ), dtype=np.float32)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        elif isinstance(random_action, dict):
            random_action = to_ndarray(random_action)
        else:
            raise TypeError(
                '`random_action` should be either int/np.ndarray or dict of int/np.ndarray, but get {}: {}'.format(
                    type(random_action), random_action
                )
            )
        return random_action


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


def get_base_manager_cfg(env_num=4):
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
            'scale': 1.0,
        } for i in range(env_num)],
        'episode_num': 2,
        'reset_timeout': 10,
        'step_timeout': 8,
        'max_retry': 5,
    }
    return EasyDict(manager_cfg)


def get_subprecess_manager_cfg(env_num=4):
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
            'scale': 1.0,
        } for i in range(env_num)],
        'episode_num': 2,
        #'step_timeout': 8,
        #'reset_timeout': 10,
        'connect_timeout': 8,
        'step_timeout': 5,
        'max_retry': 2,
    }
    return EasyDict(manager_cfg)


def get_gym_vector_manager_cfg(env_num=4):
    manager_cfg = {
        'env_cfg': [{
            'name': 'name{}'.format(i),
        } for i in range(env_num)],
        'episode_num': 2,
        'connect_timeout': 8,
        'step_timeout': 5,
        'max_retry': 2,
        'share_memory': True
    }
    return EasyDict(manager_cfg)


@pytest.fixture(scope='function')
def setup_base_manager_cfg():
    manager_cfg = get_base_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeEnv, cfg=c) for c in env_cfg]
    return deep_merge_dicts(BaseEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_fast_base_manager_cfg():
    manager_cfg = get_base_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    for e in env_cfg:
        e['scale'] = 0.1
    manager_cfg['env_fn'] = [partial(FakeEnv, cfg=c) for c in env_cfg]
    return deep_merge_dicts(BaseEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_sync_manager_cfg():
    manager_cfg = get_subprecess_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    # TODO(nyz) test fail when shared_memory = True
    manager_cfg['shared_memory'] = False
    manager_cfg['env_fn'] = [partial(FakeEnv, cfg=c) for c in env_cfg]
    return deep_merge_dicts(SyncSubprocessEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_async_manager_cfg():
    manager_cfg = get_subprecess_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeAsyncEnv, cfg=c) for c in env_cfg]
    manager_cfg['shared_memory'] = False
    return deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), EasyDict(manager_cfg))


@pytest.fixture(scope='function')
def setup_gym_vector_manager_cfg():
    manager_cfg = get_subprecess_manager_cfg(4)
    env_cfg = manager_cfg.pop('env_cfg')
    manager_cfg['env_fn'] = [partial(FakeGymEnv, cfg=c) for c in env_cfg]
    manager_cfg['shared_memory'] = False
    return EasyDict(manager_cfg)
