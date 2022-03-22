from typing import Any, List, Union, Optional
from collections import namedtuple
from easydict import EasyDict
import copy
import os
import time
import gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, ViewSizeWrapper
from gym_minigrid.window import Window

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('minigrid')
class MiniGridEnv(BaseEnv):
    config = dict(
        env_id='MiniGrid-KeyCorridorS3R3-v0',
        flat_obs=True,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._env_id = cfg.env_id
        self._flat_obs = cfg.flat_obs
        self._save_replay = False
        # self._max_step = MINIGRID_INFO_DICT[self._env_id].max_step
        self._max_step = cfg.max_step


    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            if self._env_id in ['MiniGrid-AKTDT-13x13-v0' or 'MiniGrid-AKTDT-13x13-1-v0']:
                # customize the agent field of view size, note this must be an odd number 
                # This also related to the observation space, see gym_minigrid.wrappers for more details
                self._env = ViewSizeWrapper(self._env, agent_view_size=5)  
            if self._env_id == 'MiniGrid-AKTDT-7x7-1-v0':
                self._env = ViewSizeWrapper(self._env, agent_view_size=3)
            if self._flat_obs:
                self._env = FlatObsWrapper(self._env)
                # self._env = RGBImgPartialObsWrapper(self._env)
                # self._env = ImgObsWrapper(self._env)
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        self._current_step = 0
        if self._save_replay:
            self._frames = []

        self._observation_space = self._env.observation_space
        self._observation_space.dtype = np.dtype('float32')
        self._action_space = self._env.action_space
        self._reward_space = gym.spaces.Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
        )

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
        if action.shape == (1,):
            action = action.squeeze()  # 0-dim array
        if self._save_replay:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)
        rew = float(rew)
        self._final_eval_reward += rew
        self._current_step += 1
        if self._current_step >= self._max_step:
            done = True
        if done:
            info['final_eval_reward'] = self._final_eval_reward
            info['current_step'] = self._current_step
            info['max_step'] = self._max_step
            if self._save_replay:
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self._env_id, self._save_replay_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transferred to a array with shape (1,)
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

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "DI-engine MiniGrid Env({})".format(self._cfg.env_id)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._save_replay = True
        self._replay_path = replay_path
        self._save_replay_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)
