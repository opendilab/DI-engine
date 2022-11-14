import copy
import os
from typing import Dict, Optional

import gym
import gym_hybrid
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from matplotlib import animation

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('gym_hybrid')
class GymHybridEnv(BaseEnv):
    default_env_id = ['Sliding-v0', 'Moving-v0', 'HardMove-v0']

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        replay_path=None,
        save_replay_gif=False,
        replay_path_gif=None,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._act_scale = cfg.act_scale
        self._init_flag = False
        self._replay_path = cfg.replay_path
        self._save_replay_gif = cfg.save_replay_gif
        self._replay_path_gif = cfg.replay_path_gif
        self._save_replay_count = 0
        if self._save_replay_gif:
            self._frames = []

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            if self._env_id == 'HardMove-v0':
                self._env = gym.make(self._env_id, num_actuators=self._cfg.num_actuators)
            else:
                self._env = gym.make(self._env_id)
            self._observation_space = self._env.observation_space
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
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Dict) -> BaseEnvTimestep:
        if self._act_scale:
            if self._env_id == 'HardMove-v0':
                action = [
                    action['action_type'], [affine_transform(i, min_val=-1, max_val=1) for i in action['action_args']]
                ]
            else:
                # acceleration_value.
                action['action_args'][0] = affine_transform(action['action_args'][0], min_val=0, max_val=1)
                # rotation_value. Following line can be omitted, because in the affine_transform function,
                # we have already done the clip(-1,1) operation
                action['action_args'][1] = affine_transform(action['action_args'][1], min_val=-1, max_val=1)
                action = [action['action_type'], action['action_args']]
        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        if done:
            info['final_eval_reward'] = self._final_eval_reward
            if self._save_replay_gif:
                if self._env_id == 'HardMove-v0':
                    self._env_id = f'hardmove_n{self._cfg.num_actuators}'
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self._env_id, self._save_replay_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1

        obs = to_ndarray(obs)
        if isinstance(obs, list):  # corner case
            for i in range(len(obs)):
                if len(obs[i].shape) == 0:
                    obs[i] = np.array([obs[i]])
            obs = np.concatenate(obs)
        assert isinstance(obs, np.ndarray) and obs.shape == (10, )
        obs = obs.astype(np.float32)

        rew = to_ndarray([rew])  # wrapped to be transferred to a numpy array with shape (1,)
        if isinstance(rew, list):
            rew = rew[0]
        assert isinstance(rew, np.ndarray) and rew.shape == (1, )
        info['action_args_mask'] = np.array([[1, 0], [0, 1], [0, 0]])
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> Dict:
        # action_type: 0, 1, 2
        # action_args:
        #   - acceleration_value: [0, 1]
        #   - rotation_value: [-1, 1]
        raw_action = self._action_space.sample()
        return {'action_type': raw_action[0], 'action_args': raw_action[1]}

    def __repr__(self) -> str:
        return "DI-engine gym hybrid Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

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
