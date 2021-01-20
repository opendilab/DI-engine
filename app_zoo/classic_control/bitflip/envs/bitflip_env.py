import gym
import numpy as np

from collections import OrderedDict, namedtuple
from typing import Any, Dict, Optional, Union, List

from nervex.envs import BaseEnv, register_env, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo


GoalEnvTimestep = namedtuple('GoalEnvTimestep', ['obs', 'goal', 'reward', 'done', 'info'])


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

    def step(self, action: np.ndarray) -> GoalEnvTimestep:
        self._curr_step += 1
        self._state[action] = 1 - self._state[action]
        if self.check_success(self._state, self._goal):
            rew = np.array([1]).astype(np.float32)
            done = True
        else:
            rew = np.array([0]).astype(np.float32)
            done = False
        self._final_eval_reward += rew
        if self._curr_step > self._maxsize:
            done = True
        info = {}
        if done:
            info['final_eval_reward'] = self._final_eval_reward

        obs = np.concatenate([self._state, self._goal], axis=0)

        return GoalEnvTimestep(obs, self._goal, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (4, ), {
                    'min': [-4.8, float("-inf"), -0.42, float("-inf")],
                    'max': [4.8, float("inf"), 0.42, float("inf")],
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


register_env('bitflip', BitFlipEnv)
