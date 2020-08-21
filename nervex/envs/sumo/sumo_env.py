import copy
import os
from collections import namedtuple
import sys
from sumolib import checkBinary
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.sumo.action.sumo_action_runner import SumoRawActionRunner
from nervex.envs.sumo.reward.sumo_reward_runner import SumoRewardRunner
from nervex.envs.sumo.obs.sumo_obs_runner import SumoObsRunner
import numpy as np
import yaml
from easydict import EasyDict
from nervex.utils import override, merge_dicts, pretty_print, read_config

import time
import traci
from functools import reduce


def build_config(user_config):
    """Aggregate a general config at the highest level class: Learner"""
    with open(os.path.join(os.path.dirname(__file__), 'sumo_env_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    default_config = cfg.env
    return merge_dicts(default_config, user_config)


class SumoWJ3Env(BaseEnv):
    r"""
    Overview:
        Sumo WangJing 3 intersection env
    Interface:
        __init__, reset, close, step, info
    """
    timestep = namedtuple('SumoTimestep', ['obs', 'reward', 'done', 'info'])
    info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space', 'agent_num'])

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            initialize sumo WJ 3 intersection Env
        Arguments:
            - cfg (:obj:`dict`): config, you can refer to `env/sumo/sumo_env_default_config.yaml`
        """
        cfg = build_config(cfg)
        self._cfg = cfg

        self._sumocfg_path = os.path.dirname(__file__) + '/' + cfg.sumocfg_path
        self._max_episode_steps = cfg.max_episode_steps
        self._inference = cfg.inference
        self._yellow_duration = cfg.yellow_duration
        self._green_duration = cfg.green_duration

        self._obs_helper = SumoObsRunner(cfg.obs)
        cfg.reward.tls = cfg.obs.tls
        cfg.reward.incoming_roads = cfg.obs.incoming_roads
        self._reward_helper = SumoRewardRunner(cfg.reward)
        self._action_helper = SumoRawActionRunner(cfg.action)
        self._launch_env_flag = False

    def _launch_env(self, gui=False):
        if self._launch_env_flag:
            return
        # set gui=True can get visualization simulation result with sumo, apply gui=False in the normal training
        # and test setting

        # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # setting the cmd mode or the visual mode
        if gui is False:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # setting the cmd command to run sumo at simulation time
        sumo_cmd = [
            sumoBinary, "-c", self._sumocfg_path, "--no-step-log", "true", "--waiting-time-memory",
            str(self._max_episode_steps)
        ]
        traci.start(sumo_cmd, label=time.time())
        self._launch_env_flag = True

    def reset(self):
        r"""
        Overview:
            reset the current env
        Returns:
            - obs (:obj:`torch.Size([380])`): the observation to env after reset
        """
        self._current_steps = 0
        self._launch_env()
        self._reward_helper.reset()
        self._action_helper.reset()
        obs = self._obs_helper.reset()
        return obs

    def close(self):
        r"""
        Overview:
            close the traci env
        """
        traci.close()

    def step(self, action: list) -> 'SumoWJ3Env.timestep':
        r"""
        Overview:
            step the sumo env with action
        Arguments:
            - action(:obj:`list`): list of length 3, represent 3 actions to take in 3 junction
        Returns:
            - timpstep(:obj:`SumoWJ3Env.timestep`): the timestep, contain obs(:obj:`torch.Size([380])`),
                reward(:obj:`float`),
                done(:obj:`bool`) and info(:obj:`dict`)
        """
        assert self._launch_env_flag
        self.action = action
        raw_action = self._action_helper.get(self)
        self._simulate(raw_action)

        obs = self._obs_helper.get(self)
        reward = self._reward_helper.get(self) if not self._inference else 0.
        done = self._current_steps >= self._max_episode_steps
        info = {}
        # return obs, reward, done, info
        return SumoWJ3Env.timestep(obs, reward, done, info)

    def seed(self, seed: int) -> None:
        pass

    def _simulate(self, raw_action: dict) -> None:
        for tls, v in raw_action.items():
            yellow_phase = v['yellow_phase']
            if yellow_phase is not None:
                traci.trafficlight.setPhase(tls, yellow_phase)
        traci.simulationStep(self._yellow_duration)
        self._current_steps += self._yellow_duration

        for tls, v in raw_action.items():
            green_phase = v['green_phase']
            traci.trafficlight.setPhase(tls, green_phase)
        traci.simulationStep(self._green_duration)
        self._current_steps += self._green_duration

    def info(self) -> 'SumoWJ3Env.info':
        r"""
        Overview:
            return the info_template of env
        Returns:
            - info_template(:obj:`SumoWJ3Env.info_template`):
                the info_template contain information about agent_num,
                observation space, action space and reward space.
        """
        info_data = {
            'agent_num': 1,
            'obs_space': self._obs_helper.info,
            'act_space': self._action_helper.info,
            'rew_space': self._reward_helper.info,
        }
        return SumoWJ3Env.info_template(**info_data)

    def __repr__(self) -> str:
        return 'sumoEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, _action):
        self._action = _action


SumoTimestep = SumoWJ3Env.timestep
