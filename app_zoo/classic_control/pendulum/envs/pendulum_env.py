from typing import Any
import gym
import torch
from nervex.envs import BaseEnv, register_env
from nervex.envs.common.env_element import EnvElement
from nervex.envs.common.common_function import affine_transform
from nervex.torch_utils import to_tensor


class PendulumEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._use_act_scale = cfg.use_act_scale
        self._env = gym.make('Pendulum-v0')

    def reset(self) -> torch.Tensor:
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

    def step(self, action: torch.Tensor) -> BaseEnv.timestep:
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
        T = EnvElement.info_template
        return BaseEnv.info_template(
            agent_num=1,
            obs_space=T((3, ), {
                'min': [-1.0, -1.0, -8.0],
                'max': [1.0, 1.0, 8.0]
            }, None, None),
            act_space=T((1, ), {
                'min': -2.0,
                'max': 2.0
            }, None, None),
            rew_space=T((1, ), {
                'min': -1 * (3.14 * 3.14 + 0.1 * 8 * 8 + 0.001 * 2 * 2),
                'max': -0.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX Pendulum Env({})".format(self._cfg.env_id)


register_env('pendulum', PendulumEnv)
