from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sc2learner.envs.common.const import ALLY_TYPE
from sc2learner.envs.common.const import MAP


class UnitTypeCountMapFeature(object):

    def __init__(self, type_map, resolution):
        self._type_map = type_map
        self._resolution = resolution

    def features(self, observation, need_flip=False):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
        self_features = self._generate_features(self_units)
        enemy_features = self._generate_features(enemy_units)
        features = np.concatenate((self_features, enemy_features))
        if need_flip:
            features = np.flip(np.flip(features, axis=1), axis=2).copy()
        return features

    @property
    def num_channels(self):
        return (max(self._type_map.values()) + 1) * 2

    def _generate_features(self, units):
        num_channels = max(self._type_map.values()) + 1
        features = np.zeros((num_channels, self._resolution, self._resolution),
                            dtype=np.float32)
        grid_width = (MAP.WIDTH - MAP.LEFT - MAP.RIGHT) / self._resolution
        grid_height = (MAP.HEIGHT - MAP.TOP - MAP.BOTTOM) / self._resolution
        for u in units:
            if u.unit_type in self._type_map:
                c = self._type_map[u.unit_type]
                x = (u.float_attr.pos_x - MAP.LEFT) // grid_width
                y = self._resolution - 1 - \
                    (u.float_attr.pos_y - MAP.BOTTOM) // grid_height
                features[c, int(y), int(x)] += 1.0
        return features / 5.0


class AllianceCountMapFeature(object):

    def __init__(self, resolution):
        self._resolution = resolution

    def features(self, observation, need_flip=False):
        self_units = [u for u in observation['units']
                      if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in observation['units']
                       if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
        neutral_units = [u for u in observation['units']
                         if u.int_attr.alliance == ALLY_TYPE.NEUTRAL.value]
        self_features = self._generate_features(self_units)
        enemy_features = self._generate_features(enemy_units)
        neutral_features = self._generate_features(neutral_units)
        features = np.concatenate((self_features, enemy_features, neutral_features))
        if need_flip:
            features = np.flip(np.flip(features, axis=1), axis=2).copy()
        return features

    @property
    def num_channels(self):
        return 3

    def _generate_features(self, units):
        features = np.zeros((1, self._resolution, self._resolution),
                            dtype=np.float32)
        grid_width = (MAP.WIDTH - MAP.LEFT - MAP.RIGHT) / self._resolution
        grid_height = (MAP.HEIGHT - MAP.TOP - MAP.BOTTOM) / self._resolution
        for u in units:
            x = (u.float_attr.pos_x - MAP.LEFT) // grid_width
            y = self._resolution - 1 - \
                (u.float_attr.pos_y - MAP.BOTTOM) // grid_height
            features[0, int(y), int(x)] += 1.0
        return features / 5.0
