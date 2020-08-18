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

import time
import traci
from functools import reduce


class SumoWJ3Env(BaseEnv):
    r"""
    Overview: Sumo WangJing 3 intersection env
    """
    timestep = namedtuple('SumoTimestep', ['obs', 'reward', 'done', 'info'])
    info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space', 'agent_num'])

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            initialize sumo WJ 3 intersection Env
        Arguments:
            - cfg (:obj:`dict`): config, you can refer to `env/sumo/sumo_env_default_config.yaml`
        """
        self._cfg = cfg

        self._sumocfg_path = cfg.sumocfg_path
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

        return sumo_cmd

    def reset(self):
        self._current_steps = 0
        obs = self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        self._launch_env()
        return obs

    def close(self):
        traci.close()

    def step(self, action: int) -> 'SumoWJ3Env.timestep':
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
