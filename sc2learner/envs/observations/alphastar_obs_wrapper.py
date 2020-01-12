from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import gym
from pysc2.lib.features import FeatureUnit
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
    BUFFS_REORDER, ABILITIES_REORDER, NUM_UPGRADES, UPGRADES_REORDER
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
        if len(feature_unit.shape) == 1:  # when feature_unit is None
            return None, None
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
    def __init__(self, cfg):
        self.cfg = cfg

    def parse(self, obs):
        np.save('raw', obs['raw_data'])
        ret = {}
        for idx, item in enumerate(self.cfg):
            key = item['key']
            ori = item['ori']
            item_data = obs[ori]
            item_data = torch.LongTensor(item_data)
            item_data = item['op'](item_data)
            item_data = item_data.squeeze()
            ret[key] = item_data
        return ret


class AlphastarObsWrapper(gym.Wrapper):

    def __init__(self, env, spatial_obs_cfg, entity_obs_cfg, scalar_obs_cfg):
        super(AlphastarObsWrapper, self).__init__(env)
        self.spatial_wrapper = SpatialObsWrapper(spatial_obs_cfg)
        self.entity_wrapper = EntityObsWrapper(entity_obs_cfg)
        self.scalar_wrapper = ScalarObsWrapper(scalar_obs_cfg)

    def _get_obs(self, obs):
        entity_info, entity_location = self.entity_wrapper.parse(obs)
        ret = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_location': entity_location,
        }
        #print(ret['spatial_info'].shape)
        #print(ret['entity_info'].shape)
        #print(len(ret['entity_location']))
        #for k, v in ret['scalar_info'].items():
        #    print(k, v.shape)
        return ret

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._get_obs(obs)
        return obs


class AlphastarObsParser(object):

    def __init__(self):
        self.spatial_wrapper = SpatialObsWrapper(transform_spatial_data())
        self.entity_wrapper = EntityObsWrapper(transform_entity_data())
        template_obs, template_replay = transform_scalar_data()
        self.scalar_wrapper = ScalarObsWrapper(template_obs)

    def parse(self, obs):
        entity_info, entity_location = self.entity_wrapper.parse(obs)
        ret = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_location': entity_location,
        }
        return ret


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
    try:
        for idx in range(v.shape[0]):
            new_v[idx] = dictionary[v[idx].item()]
    except KeyError as e:
        print(e, num)
        #raise KeyError
    return one_hot(new_v, num)


def div_func(inputs, other, unsqueeze_dim=1):
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)


def binary_encode(v, bit_num):
    bin_v = '{:b}'.format(int(v))
    bin_v = [int(i) for i in bin_v]
    bit_diff = len(bin_v) - bit_num
    if bit_diff > 0:
        bin_v = bin_v[-bit_num:]
    elif bit_diff < 0:
        bin_v = [0 for _ in range(-bit_diff)] + bin_v
    return torch.FloatTensor(bin_v)


def batch_binary_encode(v, bit_num):
    assert(len(v.shape) == 1)
    v = v.clamp(0)
    B = v.shape[0]
    ret = []
    for b in range(B):
        try:
            ret.append(binary_encode(v[b], bit_num))
        except ValueError:
            print('ValueError', v)
            raise ValueError
    return torch.stack(ret, dim=0)


def reorder_boolean_vector(v, dictionary, num):
    ret = torch.zeros(num)
    for item in v:
        try:
            idx = dictionary[item.item()]
        except KeyError as e:
            print(e, item, num)
            print(dictionary)
            raise KeyError
        ret[idx] = 1
    return ret


def clip_one_hot(v, num):
    v = v.clamp(0, num-1)
    return one_hot(v, num)


def transform_entity_data(resolutin=128, pad_value=-1e9):

    template = [
        {'key': 'unit_type', 'dim': NUM_UNIT_TYPES, 'op': partial(
            reorder_one_hot, dictionary=UNIT_TYPES_REORDER, num=NUM_UNIT_TYPES), 'other': 'one-hot'},
        #{'key': 'unit_attr', 'dim': 13, 'other': 'each one boolean'},
        {'key': 'alliance', 'dim': 5, 'op': partial(one_hot, num=5), 'other': 'one-hot'},
        {'key': 'health', 'dim': 39, 'op': partial(sqrt_one_hot, max_val=1500), 'other': 'one-hot, sqrt(1500), floor'},
        {'key': 'shield', 'dim': 32, 'op': partial(sqrt_one_hot, max_val=1000), 'other': 'one-hot, sqrt(1000), floor'},
        {'key': 'energy', 'dim': 15, 'op': partial(sqrt_one_hot, max_val=200), 'other': 'one-hot, sqrt(200), floor'},
        {'key': 'cargo_space_taken', 'dim': 9, 'op': partial(clip_one_hot, num=9), 'other': 'one-hot'},
        {'key': 'cargo_space_max', 'dim': 9, 'op': partial(clip_one_hot, num=9), 'other': 'one-hot'},  # 1020 ???
        {'key': 'build_progress', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'health_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'shield_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'energy_ratio', 'dim': 1, 'op': partial(div_func, other=256.), 'other': 'float [0, 1]'},
        {'key': 'display_type', 'dim': 5, 'op': partial(one_hot, num=5), 'other': 'one-hot'},
        {'key': 'x', 'dim': 8, 'op': partial(batch_binary_encode, bit_num=8), 'other': 'binary encoding'},
        {'key': 'y', 'dim': 8, 'op': partial(batch_binary_encode, bit_num=8), 'other': 'binary encoding'},
        {'key': 'cloak', 'dim': 5, 'op': partial(one_hot, num=5), 'other': 'one-hot'},
        {'key': 'is_powered', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'hallucination', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'active', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'is_on_screen', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'is_in_cargo', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'mineral_contents', 'dim': 20, 'op': partial(
            div_one_hot, max_val=1900, ratio=100), 'other': 'one-hot, 1900/100'},
        {'key': 'vespene_contents', 'dim': 27, 'op': partial(
            div_one_hot, max_val=2600, ratio=100), 'other': 'one-hot, 2600/100'},
        {'key': 'minerals', 'dim': 43, 'op': partial(
            sqrt_one_hot, max_val=1800), 'ori': 'player', 'other': 'one-hot, sqrt(1800), floor'},
        {'key': 'vespene', 'dim': 51, 'op': partial(sqrt_one_hot, max_val=2500),
         'ori': 'player', 'other': 'one-hot, sqrt(2500), floor'},
        {'key': 'assigned_harvesters', 'dim': 34, 'op': partial(one_hot, num=34), 'other': 'one-hot'},
        {'key': 'ideal_harvesters', 'dim': 18, 'op': partial(one_hot, num=18), 'other': 'one-hot'},
        {'key': 'weapon_cooldown', 'dim': 32, 'op': partial(clip_one_hot, num=32), 'other': 'one-hot, game steps'},  # 35??
        {'key': 'order_length', 'dim': 9, 'op': partial(one_hot, num=9), 'other': 'one-hot'},
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
        {'key': 'addon_unit_type', 'dim': 2, 'op': partial(one_hot, num=2), 'other': 'one-hot'},
        {'key': 'order_progress_0', 'dim': 10, 'op': partial(
            div_one_hot, max_val=1, ratio=0.1), 'other': 'one-hot(1/0.1)'},
        {'key': 'order_progress_1', 'dim': 10, 'op': partial(
            div_one_hot, max_val=1, ratio=0.1), 'other': 'one-hot(1/0.1)'},
        {'key': 'attack_upgrade_level', 'dim': 4, 'op': partial(one_hot, num=4), 'other': 'one-hot'},
        {'key': 'armor_upgrade_level', 'dim': 4, 'op': partial(one_hot, num=4), 'other': 'one-hot'},
        {'key': 'shield_upgrade_level', 'dim': 4, 'op': partial(one_hot, num=4), 'other': 'one-hot'},
        #{'key': 'was_selected', 'dim': 2, 'other': 'one-hot, last action'},
        #{'key': 'was_targeted', 'dim': 2, 'other': 'one-hot, last action'},
    ]
    return template


def transform_spatial_data():
    template = [
        #{'key': 'scattered_entities', 'other': '32 channel float'},
        {'key': 'camera', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'height_map', 'dim': 1, 'op': partial(
            div_func, other=256., unsqueeze_dim=0), 'other': 'float height_map/255'},
        {'key': 'visibility', 'dim': 4, 'op': partial(num_first_one_hot, num=4), 'other': 'one-hot 4 value'},
        {'key': 'creep', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'entity_owners', 'dim': 5, 'op': partial(num_first_one_hot, num=5), 'other': 'one-hot 5 value'},
        {'key': 'alerts', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'pathable', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
        {'key': 'buildable', 'dim': 2, 'op': partial(num_first_one_hot, num=2), 'other': 'one-hot 2 value'},
    ]
    return template


def transform_scalar_data():
    template_obs = [
        #{'key': 'agent_statistics', 'input_dim': 1, 'output_dim': 64, 'other': 'float'},
        {'key': 'race', 'arch': 'fc', 'input_dim': 5, 'output_dim': 32, 'ori': 'home_race_requested',
            'op': partial(num_first_one_hot, num=5), 'scalar_context': True, 'other': 'one-hot 5 value'},
        {'key': 'enemy_race', 'arch': 'fc', 'input_dim': 5, 'output_dim': 32, 'ori': 'away_race_requested',
            'op': partial(num_first_one_hot, num=5), 'scalar_context': True, 'other': 'one-hot 5 value'},  # TODO 10% hidden
        {'key': 'upgrades', 'arch': 'fc', 'input_dim': NUM_UPGRADES, 'output_dim': 128, 'ori': 'upgrades',
            'op': partial(reorder_boolean_vector, dictionary=UPGRADES_REORDER, num=NUM_UPGRADES), 'other': 'boolean'},
        #{'key': 'enemy_upgrades', 'arch': 'fc', 'input_dim': NUM_UPGRADES, 'output_dim': 128, 'ori': 'enemy_upgrades',
        #    'op': partial(reorder_boolean_vector, dictionary=UPGRADES_REORDER, num=NUM_UPGRADES), 'other': 'boolean'},
        {'key': 'time', 'arch': 'transformer', 'input_dim': 32, 'output_dim': 64, 'ori': 'game_loop',
            'op': partial(batch_binary_encode, bit_num=32), 'other': 'transformer'},

        # {'key': 'available_actions', 'input_dim': 1, 'output_dim': 64, 'scalar_context': True, 'other': 'boolean vector'},  # TODO
        {'key': 'unit_counts_bow', 'arch': 'fc', 'input_dim': 23, 'output_dim': 64, 'ori': 'feature_units_count', 'op': partial(sqrt_one_hot, max_val=512), 'other': 'square root'},
        #{'key': 'last_delay', 'input_dim': 128, 'output_dim': 64, 'other':  'one-hot 128 value'},
        # {'key': 'last_action_type', 'input_dims': [], 'output_dims': 128, 'other':  'one-hot xxx value(possible actions number)'},  # TODO
        # {'key': 'last_repeat_queued', 'input_dims': [], 'output_dims': 256, 'other':  'one-hot xxx value(possible arguments value numbers)'},  # TODO
    ]
    template_replay = [
        {'key': 'mmr', 'input_dim': 6, 'output_dim': 64, 'other': 'min(mmr / 1000, 6)'},
        {'key': 'cumulative_statistics', 'input_dims': [], 'output_dims': [32, 32, 32], 'scalar_context': True, 'other': 'boolean vector, split and concat'},
        {'key': 'beginning_build_order', 'scalar_context': True, 'other': 'transformer'},  # TODO
    ]
    return template_obs, template_replay
