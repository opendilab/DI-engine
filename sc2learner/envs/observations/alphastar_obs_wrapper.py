from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import gym
from pysc2.lib.features import FeatureUnit
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
        BUFFS_REORDER, ABILITIES_REORDER
from sc2learner.nn_utils import one_hot
from functools import partial


class SpatialObsWrapper(object):
    def __init__(self, cfg, use_feature_screen=True):
        self.feature_screen_id = {
            'height_map': 0,
            'visibility': 1,
            'creep': 2,
            'entity_owners': 5,
            'pathable': 24,
            'buildable': 25,
        }
        self.feature_minimap_id = {
            'height_map': 0,
            'visibility': 1,
            'creep': 2,
            'camera': 3,
            'entity_owners': 5,
            'alerts': 8,
            'pathable': 9,
            'buildable': 10,
        }
        self.cfg = cfg
        self.use_feature_screen = use_feature_screen
        self._dim = self._get_dim()

    def _parse(self, feature, idx_dict):
        ret = []
        for item in self.cfg:
            key = item['key']
            if key in idx_dict.keys():
                idx = idx_dict[key]
                data = feature[idx]
                data = torch.LongTensor(data)
                data = item['op'](data)
                ret.append(data)
        return ret

    def parse(self, obs):
        ret = []
        feature_minimap = obs['feature_minimap']
        ret.extend(self._parse(feature_minimap, self.feature_minimap_id))
        if self.use_feature_screen:
            feature_screen = obs['feature_screen']
            ret.extend(self._parse(feature_screen, self.feature_screen_id))

        return torch.cat(ret, dim=0)

    def _get_dim(self):
        dim = 0
        for item in self.cfg:
            key = item['key']
            if key in self.feature_minimap_id.keys():
                dim += item['dim']
        if self.use_feature_screen:
            for item in self.cfg:
                key = item['key']
                if key in self.feature_screen_id.keys():
                    dim += item['dim']
        self._dim = dim

    @property
    def dim(self):
        return self._dim


class EntityObsWrapper(object):
    def __init__(self, cfg, use_raw_units=False):
        self.use_raw_units = use_raw_units
        self.key = 'feature_units' if not self.use_raw_units else 'raw_units'
        self.cfg = cfg

    def parse(self, obs):
        feature_unit = obs[self.key]
        num_unit, num_attr = feature_unit.shape
        entity_location = []
        for idx in range(num_unit):
            entity_location.append((feature_unit[idx].x, feature_unit[idx].y))

        ret = []
        for idx, item in enumerate(self.cfg):
            key = item['key']
            if 'ori' in item.keys():
                ori = item['ori']
                item_data = obs[ori][key]
                item_data = [item_data for _ in range(num_unit)]
            else:
                key_index = FeatureUnit[key]
                item_data = feature_unit[:, key_index]
            item_data = torch.LongTensor(item_data)
            item_data = item['op'](item_data)
            ret.append(item_data)
        ret = list(zip(*ret))
        ret = [torch.cat(item, dim=0) for item in ret]
        return torch.stack(ret, dim=0), entity_location


class ScalarObsWrapper(object):
    def __init__(self, cfg=None):
        '''
            mmr, race, upgrade, count, time, z
        '''
        pass

    def parse(self, obs):
        pass


class AlphastarObsWrapper(gym.Wrapper):

    def __init__(self, env, spatial_obs_cfg, entity_obs_cfg):
        super(AlphastarObsWrapper, self).__init__(env)
        self.spatial_wrapper = SpatialObsWrapper(spatial_obs_cfg)
        self.entity_wrapper = EntityObsWrapper(entity_obs_cfg)
        self.scalar_wrapper = ScalarObsWrapper()

    def _get_obs(self, obs):
        entity_info, entity_location = self.entity_wrapper.parse(obs)
        print(obs.keys())
        ret = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_location': entity_location,
        }
        return ret

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._get_obs(obs)
        return obs


num_first_one_hot = partial(one_hot, num_first=True)


def sqrt_one_hot(v, max_val):
    num = int(math.sqrt(max_val)) + 1
    v = v.float()
    v = torch.floor(torch.sqrt(torch.clamp(v, 0, max_val))).long()
    return one_hot(v, num)


def div_one_hot(v, max_val, ratio):
    num = int(max_val / ratio) + 1
    v = v.float()
    v = torch.floor(torch.clamp(v, 0, max_val) / ratio).long()
    return one_hot(v, num)


def reorder_one_hot(v, dictionary, num):
    assert(len(v.shape) == 1)
    assert(isinstance(v, torch.Tensor))
    new_v = torch.zeros_like(v)
    for idx in range(v.shape[0]):
        new_v[idx] = dictionary[v[idx].item()]
    return num_first_one_hot(new_v, num)


def div_func(inputs, other, unsqueeze_dim=1):
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)


def transform_entity_data(resolutin=128, pad_value=-1e9):

    template = [
        {'key': 'unit_type', 'dim': NUM_UNIT_TYPES, 'op': partial(
            reorder_one_hot, dictionary=UNIT_TYPES_REORDER, num=NUM_UNIT_TYPES), 'other': 'one-hot'},
        #{'key': 'unit_attr', 'dim': 13, 'other': 'each one boolean'},
        {'key': 'alliance', 'dim': 5, 'op': partial(num_first_one_hot, num=5), 'other': 'one-hot'},
        {'key': 'health', 'dim': 39, 'op': partial(sqrt_one_hot, max_val=1500), 'other': 'one-hot, sqrt(1500), floor'},
        {'key': 'shield', 'dim': 32, 'op': partial(sqrt_one_hot, max_val=1000), 'other': 'one-hot, sqrt(1000), floor'},
        {'key': 'energy', 'dim': 15, 'op': partial(sqrt_one_hot, max_val=200), 'other': 'one-hot, sqrt(200), floor'},
        {'key': 'cargo_space_taken', 'dim': 9, 'op': partial(num_first_one_hot, num=9), 'other': 'one-hot'},
        {'key': 'cargo_space_max', 'dim': 9, 'op': partial(num_first_one_hot, num=9), 'other': 'one-hot'},
        {'key': 'build_progress', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'health_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'shield_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'energy_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'display_type', 'dim': 5, 'op': partial(num_first_one_hot, num=5), 'other': 'one-hot'},
        {'key': 'x', 'dim': 1, 'op': partial(div_func, other=1.), 'other': 'binary encoding'},  # doubt
        {'key': 'y', 'dim': 1, 'op': partial(div_func, other=1.), 'other': 'binary encoding'},  # doubt
        {'key': 'cloak', 'dim': 5, 'op': partial(num_first_one_hot, num=5), 'other': 'one-hot'},
        {'key': 'is_powered', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'hallucination', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'active', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'is_on_screen', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'is_in_cargo', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'mineral_contents', 'dim': 19, 'op': partial(
            div_one_hot, max_val=1900, ratio=100), 'other': 'one-hot, 1900/100'},
        {'key': 'vespene_contents', 'dim': 26, 'op': partial(
            div_one_hot, max_val=2600, ratio=100), 'other': 'one-hot, 2600/100'},
        {'key': 'minerals', 'dim': 43, 'op': partial(sqrt_one_hot, max_val=1800), 'ori': 'player', 'other': 'one-hot, sqrt(1800), floor'},
        {'key': 'vespene', 'dim': 51, 'op': partial(sqrt_one_hot, max_val=2500), 'ori': 'player', 'other': 'one-hot, sqrt(2500), floor'},
        {'key': 'assigned_harvesters', 'dim': 24, 'op': partial(num_first_one_hot, num=24), 'other': 'one-hot'},
        {'key': 'ideal_harvesters', 'dim': 17, 'op': partial(num_first_one_hot, num=17), 'other': 'one-hot'},
        {'key': 'weapon_cooldown', 'dim': 32, 'op': partial(num_first_one_hot, num=32), 'other': 'one-hot, game steps'},
        {'key': 'order_length', 'dim': 9, 'op': partial(num_first_one_hot, num=9), 'other': 'one-hot'},
        {'key': 'order_id_0', 'dim': NUM_ABILITIES, 'op': partial(
            reorder_one_hot, dictionary=ABILITIES_REORDER, num=NUM_ABILITIES), 'other': 'one-hot'},
        {'key': 'order_id_1', 'dim': NUM_ABILITIES, 'op': partial(
            reorder_one_hot, dictionary=ABILITIES_REORDER, num=NUM_ABILITIES), 'other': 'one-hot'},  # TODO only building order
        {'key': 'order_id_2', 'dim': NUM_ABILITIES, 'op': partial(
            reorder_one_hot, dictionary=ABILITIES_REORDER, num=NUM_ABILITIES), 'other': 'one-hot'},  # TODO only building order
        {'key': 'order_id_3', 'dim': NUM_ABILITIES, 'op': partial(
            reorder_one_hot, dictionary=ABILITIES_REORDER, num=NUM_ABILITIES), 'other': 'one-hot'},  # TODO only building order
        {'key': 'buff_id_0', 'dim': NUM_BUFFS, 'op': partial(
            reorder_one_hot, dictionary=BUFFS_REORDER, num=NUM_BUFFS), 'other': 'one-hot'},
        {'key': 'buff_id_1', 'dim': NUM_BUFFS, 'op': partial(
            reorder_one_hot, dictionary=BUFFS_REORDER, num=NUM_BUFFS), 'other': 'one-hot'},
        {'key': 'addon_unit_type', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot'},
        {'key': 'order_progress_0', 'dim': 10, 'op': partial(
            div_one_hot, max_val=1, ratio=0.1), 'other': 'one-hot(1/0.1)'},
        {'key': 'order_progress_1', 'dim': 10, 'op': partial(
            div_one_hot, max_val=1, ratio=0.1), 'other': 'one-hot(1/0.1)'},
        {'key': 'attack_upgrade_level', 'dim': 4, 'op': partial(num_first_one_hot, num=4), 'other': 'one-hot'},
        {'key': 'armor_upgrade_level', 'dim': 4, 'op': partial(num_first_one_hot, num=4), 'other': 'one-hot'},
        {'key': 'shield_upgrade_level', 'dim': 4, 'op': partial(num_first_one_hot, num=4), 'other': 'one-hot'},
        #{'key': 'was_selected', 'dim': 2, 'other': 'one-hot, last action'},
        #{'key': 'was_targeted', 'dim': 2, 'other': 'one-hot, last action'},
    ]
    return template


def transform_spatial_data():
    template = [
        #{'key': 'scattered_entities', 'other': '32 channel float'},
        {'key': 'camera', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'height_map', 'dim': 1, 'op': partial(div_func, other=256., unsqueeze_dim=0), 'other': 'float height_map/255'},
        {'key': 'visibility', 'dim': 4, 'op': partial(num_first_one_hot, num=4), 'other': 'one-hot 4 value'},
        {'key': 'creep', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'entity_owners', 'dim': 5, 'op': partial(num_first_one_hot, num=5), 'other': 'one-hot 5 value'},
        {'key': 'alerts', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'pathable', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'buildable', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
    ]
    return template
