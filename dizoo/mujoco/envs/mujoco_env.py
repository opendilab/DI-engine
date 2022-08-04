from typing import Any, Union, List, Optional
import copy
import numpy as np
from easydict import EasyDict
import gym
import torch

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from .mujoco_wrappers import wrap_mujoco


@ENV_REGISTRY.register('mujoco')
class MujocoEnv(BaseEnv):

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

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env()
            if self._replay_path is not None:
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix='rl-video-{}'.format(id(self))
                )

            self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
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
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')
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
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = {'min': self.action_space.low[0], 'max': self.action_space.high[0], 'dtype': np.float32}
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
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


@ENV_REGISTRY.register('mbmujoco')
class MBMujocoEnv(MujocoEnv):
    def termination_fn(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            This function determines whether each state is a terminated state.
        .. note::
            This is a collection of termination functions for mujocos used in MBPO (arXiv: 1906.08253),\
            directly copied from MBPO repo https://github.com/jannerm/mbpo/tree/master/mbpo/static.
        """
        assert len(next_obs.shape) == 2
        if self._cfg.env_id == "Hopper-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = torch.isfinite(next_obs).all(-1) \
                       * (torch.abs(next_obs[:, 1:]) < 100).all(-1) \
                       * (height > .7) \
                       * (torch.abs(angle) < .2)

            done = ~not_done
            return done
        elif self._cfg.env_id == "Walker2d-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            return done
        elif 'walker_' in self._cfg.env_id:
            torso_height = next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in self._cfg.env_id or 'walker_5' in self._cfg.env_id:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            return done
        elif self._cfg.env_id == "HalfCheetah-v3":
            done = torch.zeros_like(next_obs.sum(-1)).bool()
            return done
        elif self._cfg.env_id in ['Ant-v2', 'AntTruncatedObs-v2']:
            x = next_obs[:, 0]
            not_done = 	torch.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)
            done = ~not_done
            return done
        elif self._cfg.env_id in ['Humanoid-v2', 'HumanoidTruncatedObs-v2']:
            z = next_obs[:,0]
            done = (z < 1.0) + (z > 2.0)
            return done
        else:
            raise KeyError("not implemented env_id: {}".format(self._cfg.env_id))
