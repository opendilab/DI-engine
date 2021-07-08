from typing import Any, Union, Optional
import gym
import torch
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list


@ENV_REGISTRY.register('pendulum')
class PendulumEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._act_scale = cfg.act_scale
        self._env = gym.make('Pendulum-v0')
        self._init_flag = False
        self._replay_path = None

    def reset(self) -> torch.Tensor:
        if not self._init_flag:
            self._env = gym.make('Pendulum-v0')
            if self._replay_path is not None:
                self._env = gym.wrappers.Monitor(
                    self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
                )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if self._act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (3, ),
                {
                    'min': [-1.0, -1.0, -8.0],
                    'max': [1.0, 1.0, 8.0],
                    'dtype': np.float32,
                },
            ),
            act_space=T(
                (1, ),
                {
                    'min': -2.0,
                    'max': 2.0,
                    'dtype': np.float32
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
                    'max': -0.0,
                    'dtype': np.float32
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine Pendulum Env({})".format(self._cfg.env_id)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
