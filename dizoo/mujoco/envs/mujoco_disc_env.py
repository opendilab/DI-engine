import copy
import os
from itertools import product
from typing import Union, List, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from matplotlib import animation

from .mujoco_wrappers import wrap_mujoco


@ENV_REGISTRY.register('mujoco-disc')
class MujocoDiscEnv(BaseEnv):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        use_act_scale=False,
        delay_reward_step=0,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._delay_reward_step = cfg.delay_reward_step
        self._init_flag = False
        self._replay_path = None
        self._save_replay_gif = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(
                self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )
            self._env = gym.wrappers.RecordVideo(self._env, './videos/' + str('time()') + '/')  # time()
        if self._save_replay_gif:
            self._frames = []
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')

        # disc_to_cont: transform discrete action index to original continuous action
        self.m = self._action_space.shape[0]
        self.n = self._cfg.each_dim_disc_size
        self.K = self.n ** self.m
        self.disc_to_cont = list(product(*[list(range(self.n)) for dim in range(self.m)]))
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

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:

        # disc_to_cont: transform discrete action index to original continuous action
        action = [-1 + 2 / self.n * k for k in self.disc_to_cont[int(action)]]
        action = to_ndarray(action)

        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        if self._use_act_scale:
            action_range = {'min': self.action_space.low[0], 'max': self.action_space.high[0], 'dtype': np.float32}
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew

        if done:
            if self._save_replay_gif:
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self._cfg.env_id, self._save_replay_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
            info['final_eval_reward'] = self._final_eval_reward

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self):
        return wrap_mujoco(
            self._cfg.env_id,
            norm_obs=self._cfg.get('norm_obs', None),
            norm_reward=self._cfg.get('norm_reward', None),
            delay_reward_step=self._delay_reward_step
        )

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._save_replay = True
        self._save_replay_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)

    def random_action(self) -> np.ndarray:
        return self.action_space.sample()

    def __repr__(self) -> str:
        return "DI-engine Mujoco Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.norm_reward.use_norm = False
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
