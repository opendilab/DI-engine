from typing import Any, List, Union, Optional
import time
import gym
import logging
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray, to_list


@ENV_REGISTRY.register('maze')
class MazeEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._seed = 0
        self._init_flag = False
        self._num_levels = cfg.get('num_levels', 1)

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make('procgen:procgen-maze-v0', start_level=0, num_levels=self._num_levels)
            self._init_flag = True
        if hasattr(self, '_seed'):
            self._env.close()
            self._env = gym.make('procgen:procgen-maze-v0', start_level=self._seed, num_levels=self._num_levels)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        np.random.seed(self._seed)
        if dynamic_seed:
            logging.warning('procgen env use num_levels to control the diversity among different episodes')

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        obs = to_ndarray(obs)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        rew = rew.astype(np.float32)
        return BaseEnvTimestep(obs, rew, bool(done), info)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (3, 64, 64),
                {
                    'min': np.zeros(shape=(3, 64, 64)),
                    'max': np.ones(shape=(3, 64, 64)) * 255,
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 15,
                    'dtype': np.float32
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': float("-inf"),
                    'max': float("inf"),
                    'dtype': np.float32
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "DI-engine Maze Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._env = gym.wrappers.Monitor(
            self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
        )
