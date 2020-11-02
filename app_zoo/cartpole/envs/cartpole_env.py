from collections import namedtuple
from typing import List, Any

import gym
import torch

from nervex.envs.env.base_env import BaseEnv
from .action.cartpole_action_runner import CartpoleRawActionRunner
from .obs.cartpole_obs_runner import CartpoleObsRunner
from .reward.cartpole_reward_runner import CartpoleRewardRunner


class CartpoleEnv(BaseEnv):
    timestep = namedtuple('cartpoleTimestep', ['obs', 'reward', 'done', 'rest_lives', 'info'])

    info_template = namedtuple('CartpoleEnvInfo', ['obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg):
        self._cfg = cfg
        self.rep_prob = 0
        self._action_helper = CartpoleRawActionRunner()
        self._reward_helper = CartpoleRewardRunner()
        self._obs_helper = CartpoleObsRunner()

        if cfg != {}:
            self._game = cfg.get('game', None)
            self._mode = cfg.get('mode', None)
            self._difficulty = cfg.get('difficulty', None)
            self._obs_type = cfg.get('obs_type', None)

        self._is_gameover = False
        self._launch_env_flag = False

    def _launch_env(self):
        self._env = gym.make("CartPole-v0").unwrapped
        self._launch_env_flag = True

    def reset(self):
        if not self._launch_env_flag:
            self._launch_env()
        ret = self._env.reset()
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        return torch.from_numpy(ret).float()

    def close(self):
        self._env.close()

    def step(self, action: torch.tensor) -> 'CartpoleEnv.timestep':
        assert self._launch_env_flag
        self.action = action.item()
        raw_action = self._action_helper.get(self)
        # env step
        self.obs, self.reward, self._is_gameover, _ = self._env.step(raw_action)

        # transform obs, reward and record statistics

        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        info = {'cum_reward': self._reward_helper.cum_reward}

        return CartpoleEnv.timestep(obs=self.obs, reward=self.reward, done=self._is_gameover, rest_lives={}, info=info)

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def info(self) -> 'CartpoleEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return CartpoleEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'CartpoleEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    @property
    def cum_reward(self) -> torch.tensor:
        return self._reward_helper.cum_reward
