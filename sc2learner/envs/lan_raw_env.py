from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from pysc2.env import sc2_env
from pysc2.env import lan_sc2_env

from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation


class LanSC2RawEnv(gym.Env):
    def __init__(self, host, config_port, agent_race, step_mul=8, resolution=32, visualize_feature_map=False):
        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=resolution, feature_minimap=resolution
        )
        self._sc2_env = lan_sc2_env.LanSC2Env(
            host=host,
            config_port=config_port,
            race=sc2_env.Race[agent_race],
            step_mul=step_mul,
            agent_interface_format=agent_interface_format,
            visualize=visualize_feature_map
        )
        self.observation_space = PySC2RawObservation(self._sc2_env.observation_spec)
        self.action_space = PySC2RawAction()
        self._reseted = False

    def step(self, actions):
        assert self._reseted
        timestep = self._sc2_env.step([actions])[0]
        observation = timestep.observation
        reward = float(timestep.reward)
        done = timestep.last()
        if done:
            self._reseted = False
        info = {}
        return (observation, reward, done, info)

    def reset(self):
        timestep = self._sc2_env.reset()[0]
        observation = timestep.observation
        self._reseted = True
        return observation

    def close(self):
        self._sc2_env.close()
