from typing import Any, Union, List
import copy
import gym
from easydict import EasyDict

from CORRO.environments.make_env import make_env

from ding.torch_utils import to_ndarray, to_list
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('meta')
class MujocoEnv(BaseEnv):
    
    def __init__(self, cfg: dict) -> None:
        self._init_flag = False
        self._use_act_scale = cfg.use_act_scale
        self._cfg = cfg

    def reset(self) -> Any:
        if not self._init_flag:
            self._env = make_env(self._cfg.env_id, 1, seed=self._cfg.seed)
            self._env.observation_space.dtype = np.float32
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')
        self._eval_episode_return = 0.
        return obs
    
    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = {'min': self.action_space.low[0], 'max': self.action_space.high[0], 'dtype': np.float32}
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        obs = to_ndarray(obs).astype('float32')
        rew = to_ndarray([rew])
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(obs, rew, done, info)
    
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