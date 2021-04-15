import copy
import numpy as np

from typing import Any, Dict, Optional, Union, List

from nervex.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.utils import ENV_REGISTRY


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

    def reset(self) -> np.ndarray:
        self._curr_step = 0
        self._final_eval_reward = 0
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

    def seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        self._state[action] = 1 - self._state[action]
        if self.check_success(self._state, self._goal):
            rew = np.array([1]).astype(np.float32)
            done = True
        else:
            rew = np.array([0]).astype(np.float32)
            done = False
        self._final_eval_reward += rew
        if self._curr_step >= self._maxsize - 1:
            done = True
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        self._curr_step += 1
        obs = np.concatenate([self._state, self._goal], axis=0)

        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (2 * self._n_bits, ), {
                    'min': [0 for _ in range(self._n_bits)],
                    'max': [1 for _ in range(self._n_bits)],
                    'dtype': float,
                }, None, None
            ),
            # [min, max)
            act_space=T((self._n_bits, ), {
                'min': 0,
                'max': self._n_bits
            }, None, None),
            rew_space=T((1, ), {
                'min': 0.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX BitFlip Env({})".format('bitflip')
