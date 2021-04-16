from typing import Any, Union, List
import copy
import torch
import numpy as np

from nervex.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.envs.common.common_function import affine_transform
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from .mujoco_wrappers import wrap_mujoco
from nervex.utils import ENV_REGISTRY


@ENV_REGISTRY.register('mujoco')
class MujocoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._env = wrap_mujoco(
            cfg.env_id,
            norm_obs=cfg.get('norm_obs', None),
            norm_reward=cfg.get('norm_reward', None),
        )

    def reset(self) -> torch.FloatTensor:
        if hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: Union[torch.Tensor, np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        if self._use_act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs)
        rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def info(self) -> BaseEnvInfo:
        reward_range = self._env.reward_range
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                observation_space.shape, {
                    'min': observation_space.low.max(),
                    'max': observation_space.high.min(),
                    'dtype': np.float32
                }, None, None
            ),
            act_space=T(
                action_space.shape, {
                    'min': action_space.low.max(),
                    'max': action_space.high.min()
                }, None, None
            ),
            rew_space=T(1, {
                'min': reward_range[0],
                'max': reward_range[1]
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX Mujoco Env({})".format(self._cfg.env_id)

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
