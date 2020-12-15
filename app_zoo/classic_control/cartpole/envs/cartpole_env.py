from typing import Any, List
import gym
import torch
from nervex.envs import BaseEnv, register_env
from nervex.envs.common.env_element import EnvElement
from nervex.torch_utils import to_tensor


class CartPoleEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env = gym.make('CartPole-v0')

    def reset(self) -> torch.Tensor:
        if hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0
        obs = self._env.reset()
        obs = to_tensor(obs, torch.float)
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: torch.Tensor) -> BaseEnv.timestep:
        if action.shape == (1, ):
            action = action.squeeze()  # 0-dim tensor
        action = action.numpy()
        obs, rew, done, info = self._env.step(action)
        obs = to_tensor(obs, torch.float)
        rew = to_tensor(rew, torch.float)
        self._final_eval_reward += rew.item()
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnv.timestep(obs, rew, done, info)

    def info(self) -> BaseEnv.info_template:
        T = EnvElement.info_template
        return BaseEnv.info_template(
            agent_num=1,
            obs_space=T(
                (4, ), {
                    'min': [-4.8, float("-inf"), -0.42, float("-inf")],
                    'max': [4.8, float("inf"), 0.42, float("inf")],
                }, None, None
            ),
            # [min, max)
            act_space=T((2, ), {
                'min': 0,
                'max': 2
            }, None, None),
            rew_space=T((1, ), {
                'min': 0.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX CartPole Env({})".format(self._cfg.env_id)


register_env('cartpole', CartPoleEnv)
