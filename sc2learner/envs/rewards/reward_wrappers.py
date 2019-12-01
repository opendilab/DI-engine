from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation
from sc2learner.envs.common.const import ALLY_TYPE


class RewardShapingWrapperV1(gym.Wrapper):

  def __init__(self, env):
    super(RewardShapingWrapperV1, self).__init__(env)
    assert isinstance(env.observation_space, PySC2RawObservation)
    self._combat_unit_types = set([UNIT_TYPE.ZERG_ZERGLING.value,
                                   UNIT_TYPE.ZERG_ROACH.value,
                                   UNIT_TYPE.ZERG_HYDRALISK.value])
    self.reward_range = (-np.inf, np.inf)

  def step(self, action):
    observation, outcome, done, info = self.env.step(action)
    n_enemies, n_self_combats = self._get_unit_counts(observation)
    if n_self_combats - n_enemies > self._n_self_combats - self._n_enemies:
      reward = 1
    elif n_self_combats - n_enemies < self._n_self_combats - self._n_enemies:
      reward = -1
    else:
      reward = 0
    if not done: reward += outcome * 10
    else: reward = outcome * 10
    self._n_enemies = n_enemies
    self._n_self_combats = n_self_combats
    return observation, reward, done, info

  def reset(self, **kwargs):
    observation = self.env.reset()
    self._n_enemies, self._n_self_combats = self._get_unit_counts(observation)
    return observation

  @property
  def action_names(self):
    if not hasattr(self.env, 'action_names'): raise NotImplementedError
    return self.env.action_names

  @property
  def player_position(self):
    if not hasattr(self.env, 'player_position'): raise NotImplementedError
    return self.env.player_position

  def _get_unit_counts(self, observation):
    num_enemy_units, num_self_combat_units = 0, 0
    for u in observation['units']:
      if u.int_attr.alliance == ALLY_TYPE.ENEMY.value:
        num_enemy_units += 1
      elif u.int_attr.alliance == ALLY_TYPE.SELF.value:
        if u.unit_type in self._combat_unit_types:
          num_self_combat_units += 1
    return num_enemy_units, num_self_combat_units


class RewardShapingWrapperV2(gym.Wrapper):

  def __init__(self, env):
    super(RewardShapingWrapperV2, self).__init__(env)
    assert isinstance(env.observation_space, PySC2RawObservation)
    self._combat_unit_types = set([UNIT_TYPE.ZERG_ZERGLING.value,
                                   UNIT_TYPE.ZERG_ROACH.value,
                                   UNIT_TYPE.ZERG_HYDRALISK.value,
                                   UNIT_TYPE.ZERG_RAVAGER.value,
                                   UNIT_TYPE.ZERG_BANELING.value,
                                   UNIT_TYPE.ZERG_BROODLING.value])
    self.reward_range = (-np.inf, np.inf)

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    n_enemies, n_selves = self._get_unit_counts(observation)
    diff_selves = n_selves - self._n_selves
    diff_enemies = n_enemies - self._n_enemies
    if not done: reward += (diff_selves - diff_enemies) * 0.02
    self._n_enemies = n_enemies
    self._n_selves = n_selves
    return observation, reward, done, info

  def reset(self, **kwargs):
    observation = self.env.reset()
    self._n_enemies, self._n_selves = self._get_unit_counts(observation)
    return observation

  @property
  def action_names(self):
    if not hasattr(self.env, 'action_names'): raise NotImplementedError
    return self.env.action_names

  @property
  def player_position(self):
    if not hasattr(self.env, 'player_position'): raise NotImplementedError
    return self.env.player_position

  def _get_unit_counts(self, observation):
    num_enemy_units, num_self_units = 0, 0
    for u in observation['units']:
      if u.int_attr.alliance == ALLY_TYPE.ENEMY.value:
        if u.unit_type in self._combat_unit_types:
          num_enemy_units += 1
      elif u.int_attr.alliance == ALLY_TYPE.SELF.value:
        if u.unit_type in self._combat_unit_types:
          num_self_units += 1
    return num_enemy_units, num_self_units


class KillingRewardWrapper(gym.Wrapper):

  def __init__(self, env):
    super(KillingRewardWrapper, self).__init__(env)
    assert isinstance(env.observation_space, PySC2RawObservation)
    self.reward_range = (-np.inf, np.inf)
    self._last_kill_value = 0

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    kill_value = observation.score_cumulative[5] + \
        observation.score_cumulative[6]
    if not done:
      reward += (kill_value - self._last_kill_value) * 1e-5
    self._last_kill_value = kill_value
    return observation, reward, done, info

  def reset(self):
    observation = self.env.reset()
    kill_value = observation.score_cumulative[5] + \
        observation.score_cumulative[6]
    self._last_kill_value = kill_value
    return observation

  @property
  def action_names(self):
    if not hasattr(self.env, 'action_names'): raise NotImplementedError
    return self.env.action_names

  @property
  def player_position(self):
    if not hasattr(self.env, 'player_position'): raise NotImplementedError
    return self.env.player_position
    observation = self.env.reset()
    return observation

  @property
  def action_names(self):
    if not hasattr(self.env, 'action_names'): raise NotImplementedError
    return self.env.action_names

  @property
  def player_position(self):
    if not hasattr(self.env, 'player_position'): raise NotImplementedError
    return self.env.player_position
