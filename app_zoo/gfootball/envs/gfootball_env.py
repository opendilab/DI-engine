import gfootball
import gfootball.env as football_env

import copy
from collections import namedtuple
from typing import List, Any

import numpy as np
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from .action.gfootball_action_runner import GfootballRawActionRunner
from .obs.gfootball_obs_runner import GfootballObsRunner
from .reward.gfootball_reward_runner import GfootballRewardRunner


@ENV_REGISTRY.register('gfootball')
class GfootballEnv(BaseEnv):

    timestep = namedtuple('GfootballTimestep', ['obs', 'reward', 'done', 'info'])

    info_template = namedtuple('GFootballEnvInfo', ['obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg):
        self._cfg = cfg
        self._action_helper = GfootballRawActionRunner(cfg)
        self._reward_helper = GfootballRewardRunner(cfg)
        self._obs_helper = GfootballObsRunner(cfg)
        self.save_replay = cfg.get("save_replay", False)
        self._launch_env_flag = False
        self._launch_env()

    def _launch_env(self, gui=False):
        self._env = football_env.create_environment(
            env_name="11_vs_11_stochastic",
            representation='raw',
            stacked=False,
            logdir='./tmp/football',
            write_goal_dumps=False,
            write_full_episode_dumps=self.save_replay,
            write_video=self.save_replay,
            render=False
        )
        self._launch_env_flag = True

    def step(self, action: np.array) -> 'GfootballEnv.timestep':
        assert self._launch_env_flag
        self.agent_action = action
        action = action.item()
        # env step
        self._football_obs, self._reward_of_action, self._is_done, self._info = self._env.step(action)
        self._football_obs = self._football_obs[0]
        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)
        info = {'cum_reward': self._reward_helper.cum_reward}
        if self._is_done:
            info['final_eval_reward'] = self._reward_helper.cum_reward
        return GfootballEnv.timestep(
            obs={
                'processed_obs': self.obs,
                'raw_obs': self._football_obs
            },
            reward=self.reward,
            done=self._is_done,
            info=info
        )

    def reset(self) -> dict:
        if not self._launch_env_flag:
            self._launch_env()
        self._football_obs = self._env.reset()[0]
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self.obs = self._obs_helper.get(self)
        return {'processed_obs': self.obs, 'raw_obs': self._football_obs}

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

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.save_replay = False
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = copy.deepcopy(cfg)
        cfg.save_replay = True
        return [cfg for _ in range(evaluator_env_num)]


GfootballTimestep = GfootballEnv.timestep
