import copy
from typing import List, Union, Optional

import gym
import numpy as np
from easydict import EasyDict

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('cliffwalking')
class CliffWalkingEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = EasyDict(
            env_id='CliffWalking',
            render_mode='rgb_array',
            max_episode_steps=300,  # default max trajectory length to truncate possible infinite attempts
        )
        self._cfg.update(cfg)
        self._init_flag = False
        self._replay_path = None
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(48, ), dtype=np.float32)
        self._env = gym.make(
            "CliffWalking", render_mode=self._cfg.render_mode, max_episode_steps=self._cfg.max_episode_steps
        )
        self._action_space = self._env.action_space
        self._reward_space = gym.spaces.Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(
                "CliffWalking", render_mode=self._cfg.render_mode, max_episode_steps=self._cfg.max_episode_steps
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            dy_seed = self._seed + 100 * np.random.randint(1, 1000)
            self._env.seed(dy_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='cliffwalking-{}'.format(id(self))
            )
        obs = self._env.reset()
        obs_encode = self._encode_obs(obs)
        self._eval_episode_return = 0.
        return obs_encode

    def close(self) -> None:
        try:
            self._env.close()
            del self._env
        except:
            pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray):
            if action.shape == (1, ):
                action = action.squeeze()  # 0-dim array
            action = action.item()
        obs, reward, done, info = self._env.step(action)
        obs_encode = self._encode_obs(obs)
        self._eval_episode_return += reward
        reward = to_ndarray([reward], dtype=np.float32)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(obs_encode, reward, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def _encode_obs(self, obs) -> np.ndarray:
        onehot = np.zeros(48, dtype=np.float32)
        onehot[int(obs)] = 1
        return onehot

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
        return "DI-engine CliffWalking Env"
