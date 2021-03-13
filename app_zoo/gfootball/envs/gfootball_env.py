import gfootball
import gfootball.env as football_env

from collections import namedtuple
from typing import List, Any

import numpy as np
from nervex.envs.env.base_env import BaseEnv
from .action.gfootball_action_runner import GfootballRawActionRunner
from .obs.gfootball_obs_runner import GfootballObsRunner
from .reward.gfootball_reward_runner import GfootballRewardRunner


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

    def _launch_env(self, gui=False):
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

    def step(self, action: np.array) -> 'GfootballEnv.timestep':
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
        return GfootballEnv.timestep(obs={'processed_obs':self.obs, 'raw_obs':self._football_obs}, reward=self.reward, done=self._is_done, info=info)

    def reset(self) -> dict:
        if not self._launch_env_flag:
            self._launch_env()
        self._football_obs = self._env.reset()[0]
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self.obs = self._obs_helper.get(self)
        return {'processed_obs':self.obs, 'raw_obs':self._football_obs}

    def seed(self, seed: int) -> None:
        self._seed = seed

    def close(self) -> None:
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

    # override
    def pack(self, timesteps: List['GfootballEnv.timestep'] = None, obs: Any = None) -> 'GfootballEnv.timestep':
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

    # override
    def unpack(self, action: Any) -> List[Any]:
        return [{'action': act} for act in action]


GfootballTimestep = GfootballEnv.timestep
