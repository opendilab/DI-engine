from typing import Any, List
import torch
import copy
import numpy as np

from nervex.envs import BaseEnv, register_env
from nervex.envs.common.env_element import EnvElement
from nervex.envs.common.common_function import affine_transform
from nervex.torch_utils import to_tensor
from .mujoco_wrappers import wrap_deepmind


class MujocoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._env = wrap_deepmind(
            cfg.env_id,
            frame_stack=cfg.frame_stack,
            episode_life=cfg.is_train,
            clip_rewards=cfg.is_train,
            norm_obs=cfg.get('norm_obs', None),
            norm_reward=cfg.get('norm_reward', None),
        )

    def reset(self) -> torch.FloatTensor:
        if hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        obs = to_tensor(obs, torch.float)
        self._final_eval_reward = 0.
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: Any) -> BaseEnv.timestep:
        action = action.numpy()
        if self._use_act_scale:
            action_range = self.info().act_space.value
            action = affine_transform(action, min_val=action_range['min'], max_val=action_range['max'])
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        obs = to_tensor(obs, torch.float)
        rew = to_tensor(rew, torch.float)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnv.timestep(obs, rew, done, info)

    def info(self) -> BaseEnv.info_template:
        reward_range = self._env.reward_range
        observation_space = self._env.observation_space
        action_space = self._env.action_space
        T = EnvElement.info_template
        return BaseEnv.info_template(
            agent_num=1,
            obs_space=T(
                observation_space.shape, {
                    'min': observation_space.low.max(),
                    'max': observation_space.high.min()
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
    def create_actor_env_cfg(cfg: dict) -> List[dict]:
        actor_env_num = cfg.pop('actor_env_num', 1)
        actor_cfg = copy.deepcopy(cfg)
        actor_cfg.is_train = True
        return [cfg for _ in range(actor_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        actor_cfg = copy.deepcopy(cfg)
        actor_cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]


register_env('mujoco', MujocoEnv)
