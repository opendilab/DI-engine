from typing import Optional
import copy
import os

import gym
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.envs.common import affine_transform, save_frames_as_gif
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('carracing')
class CarRacingEnv(BaseEnv):

    config = dict(
        replay_path=None,
        save_replay_gif=False,
        replay_path_gif=None,
        action_clip=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        # env_id:CarRacing-v2
        self._env_id = cfg.env_id
        self._replay_path = None
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_gif = cfg.save_replay_gif
        self._save_replay_count = 0
        if cfg.continuous:
            self._act_scale = cfg.act_scale  # act_scale only works in continuous env
            self._action_clip = cfg.action_clip
        else:
            self._act_scale = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._cfg.env_id, continuous=self._cfg.continuous)
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )
            self._observation_space = gym.spaces.Box(
                low=np.min(self._env.observation_space.low.astype(np.float32) / 255),
                high=np.max(self._env.observation_space.high.astype(np.float32) / 255),
                shape=(
                    self._env.observation_space.shape[2], self._env.observation_space.shape[0],
                    self._env.observation_space.shape[1]
                ),
                dtype=np.float32
            )
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._eval_episode_return = 0
        obs = self._env.reset()
        obs = obs.astype(np.float32) / 255
        obs = obs.transpose(2, 0, 1)
        obs = to_ndarray(obs)
        if self._save_replay_gif:
            self._frames = []
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.item()  # 0-dim array
        if self._act_scale:
            action = affine_transform(action, action_clip=self._action_clip, min_val=-1, max_val=1)
        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)
        obs = obs.astype(np.float32) / 255
        obs = obs.transpose(2, 0, 1)
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:
                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                path = os.path.join(
                    self._replay_path_gif, '{}_episode_{}.gif'.format(self._env_id, self._save_replay_count)
                )
                save_frames_as_gif(self._frames, path)
                self._save_replay_count += 1

        obs = to_ndarray(obs)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transferred to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._save_replay_gif = True
        self._save_replay_count = 0
        # this function can lead to the meaningless result
        self._env = gym.wrappers.RecordVideo(
            self._env,
            video_folder=self._replay_path,
            episode_trigger=lambda episode_id: True,
            name_prefix='rl-video-{}'.format(id(self))
        )

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, np.ndarray):
            pass
        elif isinstance(random_action, int):
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

    def __repr__(self) -> str:
        return "DI-engine CarRacing Env"
