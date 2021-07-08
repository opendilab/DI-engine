from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import logging
import gym
import copy
import numpy as np
from namedlist import namedlist
from collections import namedtuple
from ding.utils import import_module, ENV_REGISTRY
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from .base_env import BaseEnv, BaseEnvTimestep, BaseEnvInfo


class DingEnvWrapper(BaseEnv):

    def __init__(self, env: gym.Env, cfg: dict = None) -> None:
        self._cfg = cfg
        if self._cfg is None:
            self._cfg = dict()
        self._env = env

    # override
    def reset(self) -> None:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._final_eval_reward = 0.0
        self._action_type = self._cfg.get('action_type', 'scalar')
        return obs

    # override
    def close(self) -> None:
        self._env.close()

    # override
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    # override
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ) and self._action_type == 'scalar':
            action = action.squeeze()
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transferred to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        obs_space = self._env.observation_space
        act_space = self._env.action_space
        return BaseEnvInfo(
            agent_num=1,
            obs_space=EnvElementInfo(
                shape=obs_space.shape,
                value={
                    'min': obs_space.low,
                    'max': obs_space.high,
                    'dtype': np.float32
                },
            ),
            act_space=EnvElementInfo(
                shape=(act_space.n, ),
                value={
                    'min': 0,
                    'max': act_space.n,
                    'dtype': np.float32
                },
            ),
            rew_space=EnvElementInfo(
                shape=1,
                value={
                    'min': -1,
                    'max': 1,
                    'dtype': np.float32
                },
            ),
            use_wrappers=None
        )

    def __repr__(self) -> str:
        return "DI-engine Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        # this function can lead to the meaningless result
        # disable_gym_view_window()
        self._env = gym.wrappers.Monitor(
            self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
        )
