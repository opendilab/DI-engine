import gfootball
import gfootball.env as football_env

from collections import namedtuple
from typing import List, Any

import torch
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gfootball.action.gfootball_action_runner import GfootballRawActionRunner
from nervex.envs.gfootball.obs.gfootball_obs_runner import GfootballObsRunner
from nervex.envs.gfootball.reward.gfootball_reward_runner import GfootballRewardRunner


class GfootballEnv(BaseEnv):

    timestep = namedtuple('GfootballTimestep', ['obs', 'reward', 'done', 'info'])

    info_template = namedtuple('GFootballEnvInfo', ['obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg):
        self._cfg = cfg

        self._action_helper = GfootballRawActionRunner(cfg)
        self._reward_helper = GfootballRewardRunner(cfg)
        self._obs_helper = GfootballObsRunner(cfg)
        self._launch_env_flag = False
        self._launch_env()
        # TODO

    def _launch_env(self, gui=False):
        # TODO
        self._env = football_env.create_environment(
            env_name="11_vs_11_stochastic",
            representation='raw',
            stacked=False,
            logdir='/tmp/football',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False
        )
        self._launch_env_flag = True
        # TODO

    def step(self, action: torch.tensor) -> 'GfootballEnv.timestep':
        assert self._launch_env_flag
        self.agent_action = action
        action = action.item()
        #env step
        self._football_obs, self._reward_of_action, self._is_done, self._info = self._env.step(action)
        self._football_obs = self._football_obs[0]
        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)
        info = {'cum_reward': self._reward_helper.cum_reward}
        return GfootballEnv.timestep(obs=self.obs, reward=self.reward, done=self._is_done, info=info)

    def reset(self):
        if not self._launch_env_flag:
            self._launch_env()
        self._football_obs = self._env.reset()[0]
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self.obs = self._obs_helper.get(self)
        return self.obs

    def seed(self, seed: int) -> None:
        self._seed = seed

    def pack(self):
        pass

    def close(self):
        self._env.close()

    def __repr__(self) -> str:
        return 'GfootballEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    def info(self) -> 'GfootballEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return GfootballEnv.info_template(**info_data)

    def unpack(self, action: Any):
        pass


GfootballTimestep = GfootballEnv.timestep
