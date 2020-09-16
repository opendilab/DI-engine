import copy
import os
from collections import namedtuple
import sys
from typing import List, Any

from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.cartpole.action.cartpole_action_runner import CartpoleRawAction, CartpoleRawActionRunner
from nervex.envs.gym.cartpole.reward.cartpole_reward_runner import CartpoleReward, CartpoleRewardRunner
from nervex.envs.gym.cartpole.obs.cartpole_obs_runner import CartpoleObs, CartpoleObsRunner
import numpy as np
import gym


class CartpoleEnv(BaseEnv):
    timestep = namedtuple('cartpoleTimestep', ['obs', 'reward', 'done', 'rest_lives'])

    info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space'])

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
        return ret

    def close(self):
        self._env.close()

    def step(self, action: int) -> 'CartpoleEnv.timestep':
        assert self._launch_env_flag
        self.action = action
        raw_action = self._action_helper.get(self)
        # env step
        self.obs, self.reward, self._is_gameover, _ = self._env.step(raw_action)

        # transform obs, reward and record statistics

        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        return CartpoleEnv.timestep(obs=self.obs, reward=self.reward, done=self._is_gameover, rest_lives={})

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

    def pack(self, timesteps: List['CartpoleEnv.timestep'] = None, obs: Any = None) -> 'CartpoleEnv.timestep':
        assert not (timesteps is None and obs is None)
        assert not (timesteps is not None and obs is not None)
        if timesteps is not None:
            assert isinstance(timesteps, list)
            assert isinstance(timesteps[0], tuple)
            timestep_type = type(timesteps[0])
            items = [[getattr(timesteps[i], item) for i in range(len(timesteps))] for item in timesteps[0]._fields]
            return timestep_type(*items)
        if obs is not None:
            return obs

    def unpack(self, action: Any) -> List[Any]:
        return [{'action': act} for act in action]


cartpoleTimestep = CartpoleEnv.timestep
