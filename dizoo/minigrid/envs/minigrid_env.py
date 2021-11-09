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
from gym_minigrid.wrappers import FlatObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

MiniGridEnvInfo = namedtuple(
    'MiniGridEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space', 'max_step', 'use_wrappers']
)
MINIGRID_INFO_DICT = {
    'MiniGrid-Empty-8x8-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 8,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=100,
        use_wrappers=None,
    ),
    'MiniGrid-FourRooms-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 8,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=100,
        use_wrappers=None,
    ),
    'MiniGrid-DoorKey-16x16-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 8,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=300,
        use_wrappers=None,
    ),
    'MiniGrid-KeyCorridorS3R3-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 6,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=300,
        use_wrappers=None,
    ),
    'MiniGrid-ObstructedMaze-2Dlh-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 7,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=300,
        use_wrappers=None,
    ),
    'MiniGrid-ObstructedMaze-Full-v0': MiniGridEnvInfo(
        agent_num=1,
        obs_space=EnvElementInfo(shape=(2739, ), value={
            'min': 0,
            'max': 7,
            'dtype': np.float32
        }),
        act_space=EnvElementInfo(
            shape=(1, ),
            value={
                'min': 0,
                'max': 7,  # [0, 7)
                'dtype': np.int64,
            }
        ),
        rew_space=EnvElementInfo(shape=(1, ), value={
            'min': 0,
            'max': 1,
            'dtype': np.float32
        }),
        max_step=300,
        use_wrappers=None,
    ),
}


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
        self._max_step = MINIGRID_INFO_DICT[cfg.env_id].max_step

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
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
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        if self._save_replay:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)
        # self._env.render() # TODO(pu)
        rew = float(rew)
        self._final_eval_reward += rew
        self._current_step += 1
        if self._current_step >= self._max_step:
            done = True
        if done:
            # self._save_replay_count = self._seed  # TODO(pu)
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
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> MiniGridEnvInfo:
        return MINIGRID_INFO_DICT[self._env_id]

    def __repr__(self) -> str:
        return "DI-engine MiniGrid Env"

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
