from typing import Any, Union
import gym
import torch
import numpy as np
from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.envs.common.common_function import affine_transform
from nervex.utils import ENV_REGISTRY
from nervex.torch_utils import to_tensor, to_ndarray, to_list


@ENV_REGISTRY.register('pendulum')
class PendulumEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._env = gym.make('Pendulum-v0')

    def reset(self) -> torch.Tensor:
        if hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if self._use_act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
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
                'dtype': np.float32
            }, None, None),
            rew_space=T(
                (1, ), {
                    'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
                    'max': -0.0,
                    'dtype': np.float32
                }, None, None
            ),
        )

    def __repr__(self) -> str:
        return "nerveX Pendulum Env({})".format(self._cfg.env_id)
