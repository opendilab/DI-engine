from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from pysc2.env import sc2_env

from sc2learner.envs.spaces.pysc2_raw import PySC2RawAction
from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation
from sc2learner.utils.utils import tprint


DIFFICULTIES= {
    "1": sc2_env.Difficulty.very_easy,
    "2": sc2_env.Difficulty.easy,
    "3": sc2_env.Difficulty.medium,
    "4": sc2_env.Difficulty.medium_hard,
    "5": sc2_env.Difficulty.hard,
    "6": sc2_env.Difficulty.hard,
    "7": sc2_env.Difficulty.very_hard,
    "8": sc2_env.Difficulty.cheat_vision,
    "9": sc2_env.Difficulty.cheat_money,
    "A": sc2_env.Difficulty.cheat_insane,
}


class SC2SelfplayRawEnv(gym.Env):

  def __init__(self,
               map_name,
               step_mul=8,
               resolution=32,
               disable_fog=False,
               agent_race='random',
               opponent_race='random',
               game_steps_per_episode=None,
               tie_to_lose=False,
               score_index=None,
               random_seed=None):
    self._map_name = map_name
    self._step_mul = step_mul
    self._resolution = resolution
    self._disable_fog = disable_fog
    self._agent_race = agent_race
    self._opponent_race = opponent_race
    self._game_steps_per_episode = game_steps_per_episode
    self._tie_to_lose = tie_to_lose
    self._score_index = score_index
    self._random_seed = random_seed
    self._reseted = False
    self._first_create = True

    self._sc2_env = self._safe_create_env()
    self.observation_space = PySC2RawObservation(self._sc2_env.observation_spec)
    self.action_space = PySC2RawAction()

  def step(self, actions):
    assert self._reseted
    assert len(actions) == 2
    timesteps = self._sc2_env.step(actions)
    observation = [timesteps[0].observation, timesteps[1].observation]
    reward = float(timesteps[0].reward)
    done = timesteps[0].last()
    if done:
      self._reseted = False
      if self._tie_to_lose and reward == 0:
        reward = -1.0
      tprint("Episode Done. Outcome %f" % reward)
    info = {}
    return (observation, reward, done, info)

  def reset(self):
    timesteps = self._safe_reset()
    self._reseted = True
    return [timesteps[0].observation, timesteps[1].observation]

  def _reset(self):
    if not self._first_create:
      self._sc2_env.close()
      self._sc2_env = self._create_env()
      self._first_create = False
    return self._sc2_env.reset()

  def _safe_reset(self, max_retry=10):
    for _ in range(max_retry - 1):
      try: return self._reset()
      except: pass
    return self._reset()

  def close(self):
    self._sc2_env.close()

  def _create_env(self):
    self._random_seed = (self._random_seed + 1) & 0xFFFFFFFF
    players=[sc2_env.Agent(sc2_env.Race[self._agent_race]),
             sc2_env.Agent(sc2_env.Race[self._opponent_race])]
    agent_interface_format=sc2_env.parse_agent_interface_format(
        feature_screen=self._resolution, feature_minimap=self._resolution)
    return sc2_env.SC2Env(
        map_name=self._map_name,
        step_mul=self._step_mul,
        players=players,
        agent_interface_format=agent_interface_format,
        disable_fog=self._disable_fog,
        game_steps_per_episode=self._game_steps_per_episode,
        visualize=False,
        score_index=self._score_index,
        random_seed=self._random_seed)

  def _safe_create_env(self, max_retry=10):
    for _ in range(max_retry - 1):
      try: return self._create_env()
      except: pass
    return self._create_env()
