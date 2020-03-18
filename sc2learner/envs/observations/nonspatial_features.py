from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import ABILITY_ID as ABILITY

from sc2learner.envs.common.const import ALLY_TYPE
from sc2learner.envs.common.const import COMBAT_TYPES


class PlayerFeature(object):
    def features(self, observation):
        player_features = observation["player"][1:-1].astype(np.float32)
        food_unused = player_features[3] - player_features[2]
        player_features[-1] = food_unused if food_unused >= 0 else 0
        scale = np.array([2000, 2000, 20, 20, 20, 20, 20, 20, 20], np.float32)
        scaled_features = (player_features / scale).astype(np.float32)
        log_features = np.log10(player_features + 1).astype(np.float32)

        bins_food_unused = np.zeros(10, dtype=np.float32)
        bin_id = int((max(food_unused, 0) - 1) // 3 + 1) if food_unused <= 27 else 9
        bins_food_unused[bin_id] = 1
        return np.concatenate((scaled_features, log_features, bins_food_unused))

    @property
    def num_dims(self):
        return 9 * 2 + 10


class ScoreFeature(object):
    def features(self, observation):
        score_features = observation.score_cumulative[3:].astype(np.float32)
        score_features /= 3000.0
        log_features = np.log10(score_features + 1).astype(np.float32)
        return np.concatenate((score_features, log_features))

    @property
    def num_dims(self):
        return 10 * 2


class UnitTypeCountFeature(object):
    def __init__(self, type_list, use_regions=False):
        self._type_list = type_list
        if use_regions:
            self._regions = [
                (0, 0, 200, 176), (0, 88, 80, 176), (80, 88, 120, 176), (120, 88, 200, 176), (0, 55, 80, 88),
                (80, 55, 120, 88), (120, 55, 200, 88), (0, 0, 80, 55), (80, 0, 120, 55), (120, 0, 200, 55)
            ]
        else:
            self._regions = [(0, 0, 200, 176)]
        self._regions_flipped = [self._regions[0]] + [self._regions[10 - i] for i in range(1, len(self._regions))]

    def features(self, observation, need_flip=False):
        feature_list = []
        for region in (self._regions if not need_flip else self._regions_flipped):
            units_in_region = [u for u in observation['units'] if self._is_in_region(u, region)]
            feature_list.append(self._generate_features(units_in_region))
        return np.concatenate(feature_list)

    @property
    def num_dims(self):
        return len(self._type_list) * len(self._regions) * 2 * 2

    def _generate_features(self, units):
        self_units = [u for u in units if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in units if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
        self_features = self._get_counts(self_units)
        enemy_features = self._get_counts(enemy_units)
        features = np.concatenate((self_features, enemy_features))

        scaled_features = features / 20
        log_features = np.log10(features + 1)

        return np.concatenate((scaled_features, log_features))

    def _get_counts(self, units):
        count = {t: 0 for t in self._type_list}
        for u in units:
            if u.unit_type in count:
                count[u.unit_type] += 1
        return np.array([count[t] for t in self._type_list], dtype=np.float32)

    def _is_in_region(self, unit, region):
        return (
            unit.float_attr.pos_x >= region[0] and unit.float_attr.pos_x < region[2]
            and unit.float_attr.pos_y >= region[1] and unit.float_attr.pos_y < region[3]
        )


class UnitStatCountFeature(object):
    def __init__(self, use_regions=False):
        if use_regions:
            self._regions = [
                (0, 0, 200, 176), (0, 88, 80, 176), (80, 88, 120, 176), (120, 88, 200, 176), (0, 55, 80, 88),
                (80, 55, 120, 88), (120, 55, 200, 88), (0, 0, 80, 55), (80, 0, 120, 55), (120, 0, 200, 55)
            ]
        else:
            self._regions = [(0, 0, 200, 176)]
        self._regions_flipped = [self._regions[0]] + [self._regions[10 - i] for i in range(1, len(self._regions))]

    def features(self, observation, need_flip=False):
        feature_list = []
        for region in (self._regions if not need_flip else self._regions_flipped):
            units_in_region = [u for u in observation['units'] if self._is_in_region(u, region)]
            feature_list.append(self._generate_features(units_in_region))
        return np.concatenate(feature_list)

    @property
    def num_dims(self):
        return len(self._regions) * 2 * 4 * 2

    def _generate_features(self, units):
        self_units = [u for u in units if u.int_attr.alliance == ALLY_TYPE.SELF.value]
        enemy_units = [u for u in units if u.int_attr.alliance == ALLY_TYPE.ENEMY.value]
        self_combats = [u for u in self_units if u.unit_type in COMBAT_TYPES]
        enemy_combats = [u for u in enemy_units if u.unit_type in COMBAT_TYPES]
        self_air_units = [u for u in self_units if u.bool_attr.is_flying]
        enemy_air_units = [u for u in enemy_units if u.bool_attr.is_flying]
        self_ground_units = [u for u in self_units if not u.bool_attr.is_flying]
        enemy_ground_units = [u for u in enemy_units if not u.bool_attr.is_flying]

        features = np.array(
            [
                len(self_units),
                len(self_combats),
                len(self_ground_units),
                len(self_air_units),
                len(enemy_units),
                len(enemy_combats),
                len(enemy_ground_units),
                len(enemy_air_units)
            ],
            dtype=np.float32
        )

        scaled_features = features / 20
        log_features = np.log10(features + 1)
        return np.concatenate((scaled_features, log_features))

    def _is_in_region(self, unit, region):
        return (
            unit.float_attr.pos_x >= region[0] and unit.float_attr.pos_x < region[2]
            and unit.float_attr.pos_y >= region[1] and unit.float_attr.pos_y < region[3]
        )


class GameProgressFeature(object):
    def features(self, observation):
        game_loop = observation["game_loop"][0]
        features_60 = self._onehot(game_loop, 60)
        features_20 = self._onehot(game_loop, 20)
        features_8 = self._onehot(game_loop, 8)
        features_4 = self._onehot(game_loop, 4)

        return np.concatenate([features_60, features_20, features_8, features_4])

    def _onehot(self, value, n_bins):
        bin_width = 24000 // n_bins
        features = np.zeros(n_bins, dtype=np.float32)
        idx = int(value // bin_width)
        idx = n_bins - 1 if idx >= n_bins else idx
        features[idx] = 1.0
        return features

    @property
    def num_dims(self):
        return 60 + 20 + 8 + 4


class ActionSeqFeature(object):
    def __init__(self, n_dims_action_space, seq_len):
        self._action_seq = [-1] * seq_len
        self._n_dims_action_space = n_dims_action_space

    def reset(self):
        self._action_seq = [-1] * len(self._action_seq)

    def push_action(self, action):
        self._action_seq.pop(0)
        self._action_seq.append(action)

    def features(self):
        features = np.zeros(self._n_dims_action_space * len(self._action_seq), dtype=np.float32)
        for i, action in enumerate(self._action_seq):
            assert action < self._n_dims_action_space
            if action >= 0:
                features[i * self._n_dims_action_space + action] = 1.0
        return features

    @property
    def num_dims(self):
        return self._n_dims_action_space * len(self._action_seq)


class WorkerFeature(object):
    def features(self, dc):
        extractor_tags = set(u.tag for u in dc.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value))
        workers = dc.units_of_type(UNIT_TYPE.ZERG_DRONE.value)
        harvest_workers = [
            u for u in workers if (len(u.orders) > 0 and u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value)
        ]
        gas_workers = [u for u in harvest_workers if u.orders[0].target_tag in extractor_tags]
        mineral_workers = [u for u in harvest_workers if u.orders[0].target_tag not in extractor_tags]
        return np.array(
            [len(gas_workers),
             len(mineral_workers),
             len(workers) - len(gas_workers) - len(mineral_workers)],
            dtype=np.float32
        ) / 20.0

    @property
    def num_dims(self):
        return 3
