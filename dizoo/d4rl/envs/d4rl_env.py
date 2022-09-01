from typing import Any, Union, List
import copy
import numpy as np
import gym
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from .d4rl_wrappers import wrap_d4rl
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('d4rl')
class D4RLEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = self._make_env(only_info=False)
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
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = {'min': self.action_space.low[0], 'max': self.action_space.high[0], 'dtype': np.float32}
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs).astype('float32')
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def _make_env(self, only_info=False):
        return wrap_d4rl(
            self._cfg.env_id,
            norm_obs=self._cfg.get('norm_obs', EasyDict(use_norm=False, offline_stats=dict(use_offline_stats=False, )),),
            norm_reward=self._cfg.get('norm_reward', EasyDict(use_norm=False, )),
            only_info=only_info
        )

    def __repr__(self) -> str:
        return "DI-engine D4RL Env({})".format(self._cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        evaluator_cfg.get('norm_reward', EasyDict(use_norm=False, )).use_norm = False
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
