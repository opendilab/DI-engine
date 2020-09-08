import copy
import os
from collections import namedtuple

from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pendulum.action.pendulum_action_runner import PendulumRawAction, PendulumRawActionRunner
from nervex.envs.gym.pendulum.reward.pendulum_reward_runner import PendulumReward, PendulumRewardRunner
from nervex.envs.gym.pendulum.obs.pendulum_obs_runner import PendulumObs, PendulumObsRunner
import numpy as np
import gym


class PendulumEnv(BaseEnv):
    timestep = namedtuple('pendulumTimestep', ['obs', 'reward', 'done', 'rest_lives'])

    info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space', 'frame_skip'])

    # frame_skip: how many frame in one step, should be 1 or 2 or 4.

    def __init__(self, cfg):
        self._cfg = cfg
        self.frameskip = 1
        self.rep_prob = 0
        self._action_helper = PendulumRawActionRunner()
        self._reward_helper = PendulumRewardRunner()
        self._obs_helper = PendulumObsRunner()

        if cfg != {}:
            self._game = cfg.get('game', None)
            self._mode = cfg.get('mode', None)
            self._difficulty = cfg.get('difficulty', None)
            self._obs_type = cfg.get('obs_type', None)
            self.frameskip = cfg.get('frameskip', 1)

        self._is_gameover = False
        self._launch_env_flag = False

    def _launch_env(self):
        self._env = gym.make("Pendulum-v0").unwrapped
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

    def step(self, action: float) -> 'PendulumEnv.timestep':
        assert self._launch_env_flag
        self.action = action
        raw_action = self._action_helper.get(self)
        # env step
        self.obs, self.reward, self._is_gameover, _ = self._env.step(raw_action)

        # transform obs, reward and record statistics

        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        return PendulumEnv.timestep(obs=self.obs, reward=self.reward, done=self._is_gameover, rest_lives={})

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def info(self) -> 'PendulumEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
            'frame_skip': self.frameskip
        }
        return PendulumEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'PendulumEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))


pendulumTimestep = PendulumEnv.timestep
