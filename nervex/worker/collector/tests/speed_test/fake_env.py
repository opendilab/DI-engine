from typing import Any, List, Union, Optional
import time
import gym
import torch
import numpy as np

from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_ndarray

from nervex.worker.collector.tests.speed_test.utils import random_change


global env_sum
env_sum = 0


def env_sleep(duration):
    time.sleep(duration)
    global env_sum
    env_sum += duration


class FakeEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._obs_dim = cfg.get('obs_dim', 8)
        self._action_dim = cfg.get('action_dim', 2)
        self._episode_step_base = cfg.get('episode_step', 200)
        self._reset_time = cfg.get('reset_time', 0.)
        self._step_time = cfg.get('step_time', 0.)
        self.reset()

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed'):
            self.seed()
        self._episode_step = int(random_change(self._episode_step_base))
        env_sleep(random_change(self._reset_time))
        self._step_count = 0
        self._final_eval_reward = 0
        obs = np.random.randn(self._obs_dim)
        return obs

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        env_sleep(random_change(self._step_time))
        self._step_count += 1
        obs = np.random.randn(self._obs_dim)
        rew = np.random.randint(2)
        done = True if self._step_count == self._episode_step else False
        info = {}
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T((self._obs_dim, ), {
                'dtype': float,
            }, None, None),
            # [min, max)
            act_space=T((self._action_dim, ), {
                'min': 0,
                'max': 2
            }, None, None),
            rew_space=T((1, ), {
                'min': 0.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX CartPole Env"
