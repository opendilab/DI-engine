import copy
import random
import numpy as np
import gym
from typing import Any, Dict, Optional, Union, List

from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray


@ENV_REGISTRY.register('bitflip')
class BitFlipEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._n_bits = cfg.n_bits
        self._state = np.zeros(self._n_bits)
        self._goal = np.zeros(self._n_bits)
        self._curr_step = 0
        self._maxsize = self._n_bits
        self._final_eval_reward = 0
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(2 * self._n_bits, ), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(self._n_bits)
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self._curr_step = 0
        self._final_eval_reward = 0
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            random_seed = 100 * random.randint(1, 1000)
            np.random.seed(self._seed + random_seed)
        elif hasattr(self, '_seed'):
            np.random.seed(self._seed)
        self._state = np.random.randint(0, 2, size=(self._n_bits, )).astype(np.float32)
        self._goal = np.random.randint(0, 2, size=(self._n_bits, )).astype(np.float32)

        while (self._state == self._goal).all():
            self._goal = np.random.randint(0, 2, size=(self._n_bits, )).astype(np.float32)

        obs = np.concatenate([self._state, self._goal], axis=0)
        return obs

    def close(self) -> None:
        pass

    def check_success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return (self._state == self._goal).all()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        self._state[action] = 1 - self._state[action]
        if self.check_success(self._state, self._goal):
            rew = np.array([1]).astype(np.float32)
            done = True
        else:
            rew = np.array([0]).astype(np.float32)
            done = False
        self._final_eval_reward += float(rew)
        if self._curr_step >= self._maxsize - 1:
            done = True
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        self._curr_step += 1
        obs = np.concatenate([self._state, self._goal], axis=0)
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine BitFlip Env({})".format('bitflip')
