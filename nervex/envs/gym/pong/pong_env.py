import copy
import os
from collections import namedtuple

from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pong.action.pong_action_runner import PongRawAction, PongRawActionRunner
from nervex.envs.gym.pong.reward.pong_reward_runner import PongReward, PongRewardRunner
from nervex.envs.gym.pong.obs.pong_obs_runner import PongObs, PongObsRunner
import numpy as np
import gym


class PongEnv(BaseEnv):
    timestep = namedtuple('pongTimestep', ['obs', 'reward', 'done', 'rest_lives'])
    info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space', 'frame_skip', 'rep_prob'])

    def __init__(self, cfg):
        self._cfg = cfg
        self.frameskip = 4
        self.rep_prob = 0.25
        self._action_helper = PongRawActionRunner()
        self._reward_helper = PongRewardRunner()
        self._obs_helper = PongObsRunner()
        
        #cfg TODO
        #util_ezpickle?
        if(cfg != {}): 
            self._game = cfg.get('game', None)
            self._mode = cfg.get('mode', None)
            self._difficulty = cfg.get('difficulty', None)
            self._obs_type = cfg.get('obs_type', None)
            self.frameskip = cfg.get('frameskip', None)
            self.rep_prob = cfg.get('rep_prob', None)
        
        #TODO
        self._isGameover = False
        
        self._launch_env_flag = False
        self._env = gym.make("PongNoFrameskip-v0").unwrapped
        self._launch_env_flag = True
        # self._env.render()
        self._pong_obs = self._env.reset()
        
    def _launch_env(self):
        self._env = gym.make("PongNoFrameskip-v0").unwrapped
        self._launch_env_flag = True

    def reset(self):
        if not self._launch_env_flag:
            self._launch_env()
        ret = self._env.reset().transpose((2, 0, 1))
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        return ret

    def close(self):
        self._env.close()

    def step(self, action: int) -> 'PongrEnv.timestep':
        assert self._launch_env_flag
        self.agent_action = action

        #env step
        self._pong_obs, self._reward_of_action, self._is_gameover, self._rest_life = self._env.step(action)
        self._pong_obs = self._pong_obs.transpose((2, 0, 1))

        #transform obs, reward and record statistics

        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        return PongEnv.timestep(
            obs=self.obs,
            reward=self.reward,
            done=self._is_gameover,
            rest_lives=self._rest_life
        )

    def seed(self, seed: int) -> None:
        self._env.seed(seed)
    
    def info(self) -> 'PongEnv.info':
        info_data = {
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
            'frame_skip': self.frameskip,
            'rep_prob': self.rep_prob
        }
        return PongEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'PongEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    @property
    def agent_action(self) -> int:
        return self._agent_action

    @agent_action.setter
    def agent_action(self, _agent_action) -> None:
        self._agent_action = _agent_action

    @property
    def reward_of_action(self) -> float:
        return self._reward_of_action

    @reward_of_action.setter
    def reward_of_action(self, _reward_of_action) -> None:
        self._reward_of_action = _reward_of_action

    @property
    def pong_obs(self) -> np.ndarray:
        return self._pong_obs

    @pong_obs.setter
    def pong_obs(self, _obs) -> None:
        self._pong_obs = _obs

pongTimestep = PongEnv.timestep