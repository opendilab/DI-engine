from typing import Any, List, Union, Optional
import time
import gym
import torch
import numpy as np
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list

from nervex.worker.actor.tests.speed_test.utils import random_change


class FakeEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._obs_dim = cfg.get('obs_dim', 8)
        self._action_dim = cfg.get('action_dim', 2)
        self._episode_step_base = cfg.get('episode_step', 200)
        self._reset_time = cfg.get('reset_time', 0.)
        self._step_time = cfg.get('step_time', 0.)
        self.reset()

    def reset(self) -> torch.Tensor:
        self._episode_step = int(random_change(self._episode_step_base))
        time.sleep(random_change(self._reset_time))
        self._step_count = 0
        self._final_eval_reward = 0
        obs = np.random.randn(self._obs_dim)
        return obs

    def close(self) -> None:
        pass

    def seed(self, seed: int) -> None:
        pass

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        time.sleep(random_change(self._step_time))
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
            obs_space=T(
                (self._obs_dim, ), {
                    'dtype': float,
                }, None, None
            ),
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
