'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. parse numpy arrays observations into tensors that pytorch can use
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import gym
from pysc2.lib.features import FeatureUnit
from pysc2.lib.action_dict import ACT_TO_GENERAL_ACT, ACT_TO_GENERAL_ACT_ARRAY
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
     UNIT_TYPES_REORDER_ARRAY, BUFFS_REORDER_ARRAY, ABILITIES_REORDER_ARRAY, NUM_UPGRADES, UPGRADES_REORDER,\
     UPGRADES_REORDER_ARRAY, NUM_ACTIONS, ACTIONS_REORDER_ARRAY, ACTIONS_REORDER, NUM_ADDON, ADDON_REORDER_ARRAY,\
     NUM_BEGIN_ACTIONS, NUM_UNIT_BUILD_ACTIONS, NUM_EFFECT_ACTIONS, NUM_RESEARCH_ACTIONS,\
     UNIT_BUILD_ACTIONS_REORDER_ARRAY, EFFECT_ACTIONS_REORDER_ARRAY, RESEARCH_ACTIONS_REORDER_ARRAY,\
     BEGIN_ACTIONS_REORDER_ARRAY, NUM_ORDER_ACTIONS, ORDER_ACTIONS_REORDER_ARRAY
from sc2learner.torch_utils import one_hot
from sc2learner.envs.actions import get_available_actions_raw_data
from functools import partial, lru_cache
from collections import OrderedDict

LOCATION_BIT_NUM = 10
DELAY_BIT_NUM = 6


def compute_denominator(x):
    x = x // 2 * 2
    x = torch.div(x, 64.)
    x = torch.pow(10000., x)
    x = torch.div(1., x)
    return x


POSITION_ARRAY = compute_denominator(torch.arange(0, 64, dtype=torch.float))


def get_postion_vector(x):
    v = torch.zeros(64, dtype=torch.float)
    v[0::2] = torch.sin(x * POSITION_ARRAY[0::2])
    v[1::2] = torch.cos(x * POSITION_ARRAY[1::2])
    return v


class SpatialObsWrapper(object):
    '''
        Overview: parse spatial observation into tensors
        Interface: __init__, parse
    '''
    def __init__(self, cfg, use_feature_screen=False):
        '''
            Overview: initial related attributes
            Arguments:
                - cfg (:obj:'list'): wrapper config
                - use_feature (:obj:'bool'): whether to use screen feature
        '''
        self.feature_screen_id = {
            'height_map': 0,
            'visibility': 1,
            'creep': 2,
            'entity_owners': 5,
            'effects': 16,
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
        '''
            Overview: find corresponding setting in cfg, parse the feature
            Arguments:
                - feature (:obj:'ndarray'): the feature to parse
                - idx_dict (:obj:'dict'): feature index
            Returns:
                - ret (:obj'LongTensor'): parse result tensor
        '''
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
        '''
            Overview: gather parse results from different feature, concatenate them
            Arguments:
                - obs (:obj:'ndarray'): observation
            Returns:
                - (:obj'LongTensor'): feature tensor
        '''
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
        return dim

    @property
    def dim(self):
        return self._dim


class EntityObsWrapper(object):
    def __init__(self, cfg, use_raw_units=True):
        self.use_raw_units = use_raw_units
        self.key = 'feature_units' if not self.use_raw_units else 'raw_units'
        self.cfg = cfg

    def parse(self, obs):
        feature_unit = obs[self.key]
        if len(feature_unit.shape) == 1:  # when feature_unit is None
            return None, None
        num_unit, num_attr = feature_unit.shape
        entity_raw = {'location': [], 'id': [], 'type': []}
        for idx in range(num_unit):
            entity_raw['location'].append((feature_unit[idx].y, feature_unit[idx].x))
            entity_raw['id'].append(int(feature_unit[idx].tag))
            entity_raw['type'].append(int(feature_unit[idx].unit_type))

        ret = []
        for idx, item in enumerate(self.cfg):
            key = item['key']
            key_index = FeatureUnit[key]
            item_data = feature_unit[:, key_index]
            item_data = torch.LongTensor(item_data)
            try:
                item_data = item['op'](item_data)
            except KeyError as e:
                print('key', key)
                print(e)
                raise KeyError
            ret.append(item_data)
        ret = list(zip(*ret))
        ret = [torch.cat(item, dim=0) for item in ret]
        return torch.stack(ret, dim=0), entity_raw


class ScalarObsWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def _parse_agent_statistics(self, obs):
        player = obs['player'][1:]
        data = torch.FloatTensor(player)
        return torch.log(data + 1)

    def _parse_unit_counts(self, obs, max_val=225):
        data = obs['unit_counts']
        ret = torch.zeros(NUM_UNIT_TYPES)
        key = data.keys()
        val = list(data.values())
        val = np.sqrt(np.clip(np.array(val), 0, max_val))
        for k, v in zip(key, val):
            idx = UNIT_TYPES_REORDER[k]
            ret[idx] = v
        return ret

    def parse(self, obs):
        ret = {}
        for idx, item in enumerate(self.cfg):
            key = item['key']
            if key == 'agent_statistics':
                ret[key] = self._parse_agent_statistics(obs)
            elif key == 'unit_counts_bow':
                ret[key] = self._parse_unit_counts(obs)
            elif key in ['available_actions', 'enemy_upgrades']:
                continue  # will be parsed by an additional function
            else:
                ori = item['ori']
                item_data = obs[ori]
                item_data = torch.LongTensor(item_data)
                item_data = item['op'](item_data)
                item_data = item_data.squeeze()
                ret[key] = item_data
        return ret


class AlphastarObsParser(object):
    '''
        Overview: parse observation into tensors
        Interface: __init__, parse, merge_action
    '''
    def __init__(self):
        '''
            Overview: initial sub-parsers and related attributes
        '''
        self.spatial_wrapper = SpatialObsWrapper(transform_spatial_data())
        self.entity_wrapper = EntityObsWrapper(transform_entity_data())
        template_obs, template_replay, template_act = transform_scalar_data()
        self.scalar_wrapper = ScalarObsWrapper(template_obs)
        self.template_act = template_act
        self.repeat_action_type = -1
        self.repeat_count = 0

    def parse(self, obs):
        '''
            Overview: gather results from sub-parsers, make them a dict
            Arguments:
                - obs (:obj:'ndarray'): observation
            Returns:
                - ret (:obj'dict'): a dict includes parse results
        '''
        entity_info, entity_raw = self.entity_wrapper.parse(obs)
        ret = {
            'scalar_info': self.scalar_wrapper.parse(obs),
            'spatial_info': self.spatial_wrapper.parse(obs),
            'entity_info': entity_info,
            'entity_raw': entity_raw,
        }
        return ret

    def merge_action(self, obs, last_action, create_entity_dim=True):
        '''
            Overview: merge last action into observation, make observation compelete
            Arguments:
                - obs (:obj:'dict'): observation
                - last_action (:obj:'dict'): a dict includes last action information
                - create_entity_dim (:obj:'bool'): whether create eneity dim
            Returns:
                - obs (:obj'dict'): merged observation
        '''
        N = obs['entity_info'].shape[0]
        last_action_type = last_action['action_type']
        last_delay = last_action['delay']
        last_queued = last_action['queued']
        last_queued = last_queued if isinstance(last_queued, torch.Tensor) else torch.LongTensor([2])  # 2 as 'none'
        obs['scalar_info']['last_delay'] = self.template_act[0]['op'](torch.LongTensor(last_delay)).squeeze()
        obs['scalar_info']['last_queued'] = self.template_act[1]['op'](torch.LongTensor(last_queued)).squeeze()
        obs['scalar_info']['last_action_type'] = self.template_act[2]['op'](torch.LongTensor(last_action_type)
                                                                            ).squeeze()

        if self.repeat_action_type == last_action_type.item():
            self.repeat_count += 1
        else:
            self.repeat_action_type = last_action_type.item()
            self.repeat_count = 0
        repeat_tensor = clip_one_hot(torch.tensor([self.repeat_count]), 7).squeeze()
        obs['scalar_info']['last_queued'] = torch.cat((obs['scalar_info']['last_queued'], repeat_tensor), dim=0)

        if obs['entity_info'] is None:
            return obs

        selected_units = last_action['selected_units']
        target_units = last_action['target_units']
        if create_entity_dim:
            obs['entity_info'] = torch.cat([obs['entity_info'], torch.empty(N, 4)], dim=1)
        selected_units = selected_units.numpy() if isinstance(selected_units, torch.Tensor) else []
        obs['entity_info'][:, -3] = 0
        obs['entity_info'][:, -4] = 1
        ids_tensor = np.array(obs['entity_raw']['id'])
        for v in selected_units:
            selected = (ids_tensor == v)
            obs['entity_info'][selected, -3] = 1
            obs['entity_info'][selected, -4] = 0

        target_units = target_units.numpy() if isinstance(target_units, torch.Tensor) else []
        obs['entity_info'][:, -1] = 0
        obs['entity_info'][:, -2] = 1
        for v in target_units:
            targeted = (ids_tensor == v)
            obs['entity_info'][targeted, -1] = 1
            obs['entity_info'][targeted, -2] = 0
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


def reorder_one_hot(v, dictionary, num, transform=None):
    assert (len(v.shape) == 1)
    assert (isinstance(v, torch.Tensor))
    new_v = torch.zeros_like(v)
    for idx in range(v.shape[0]):
        if transform is None:
            val = v[idx].item()
        else:
            val = transform[v[idx].item()]
        new_v[idx] = dictionary[val]
    return one_hot(new_v, num)


def reorder_one_hot_array(v, array, num, transform=None):
    v = v.numpy()
    if transform is None:
        val = array[v]
    else:
        val = array[transform[v]]
    return one_hot(torch.LongTensor(val), num)


def div_func(inputs, other, unsqueeze_dim=1):
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)


@lru_cache(maxsize=32)
def get_to_and(num_bits):
    return 2**np.arange(num_bits - 1, -1, -1).reshape([1, num_bits])


def batch_binary_encode(x, bit_num):
    # Big endian binary encode to float tensor
    # Example: >>> batch_binary_encode(torch.tensor([131,71]), 10)
    # tensor([[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
    #         [0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]])
    x = x.numpy()
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = get_to_and(bit_num)
    return torch.FloatTensor((x & to_and).astype(bool).astype(float).reshape(xshape + [bit_num]))


def reorder_boolean_vector(v, dictionary, num, transform=None):
    ret = torch.zeros(num)
    for item in v:
        try:
            if transform is None:
                val = item.item()
            else:
                val = transform[item.item()]
            idx = dictionary[val]
        except KeyError as e:
            # print(dictionary)
            raise KeyError('{}_{}_'.format(num, e))
        ret[idx] = 1
    return ret


def clip_one_hot(v, num):
    v = v.clamp(0, num - 1)
    return one_hot(v, num)


def transform_entity_data(resolution=128, pad_value=-1e9):

    template = [
        {
            'key': 'build_progress',
            'dim': 1,
            'op': partial(div_func, other=100.),
            'other': 'float [0, 1]'
        },
        {
            'key': 'health_ratio',
            'dim': 1,
            'op': partial(div_func, other=255.),
            'other': 'float [0, 1]'
        },
        {
            'key': 'shield_ratio',
            'dim': 1,
            'op': partial(div_func, other=255.),
            'other': 'float [0, 1]'
        },
        {
            'key': 'energy_ratio',
            'dim': 1,
            'op': partial(div_func, other=255.),
            'other': 'float [0, 1]'
        },
        {
            'key': 'unit_type',
            'dim': NUM_UNIT_TYPES,
            'op': partial(reorder_one_hot_array, array=UNIT_TYPES_REORDER_ARRAY, num=NUM_UNIT_TYPES),
            'other': 'one-hot'
        },
        #{'key': 'unit_attr', 'dim': 13, 'other': 'each one boolean'},
        {
            'key': 'alliance',
            'dim': 5,
            'op': partial(one_hot, num=5),
            'other': 'one-hot'
        },
        {
            'key': 'health',
            'dim': 39,
            'op': partial(sqrt_one_hot, max_val=1500),
            'other': 'one-hot, sqrt(1500), floor'
        },
        {
            'key': 'shield',
            'dim': 32,
            'op': partial(sqrt_one_hot, max_val=1000),
            'other': 'one-hot, sqrt(1000), floor'
        },
        {
            'key': 'energy',
            'dim': 15,
            'op': partial(sqrt_one_hot, max_val=200),
            'other': 'one-hot, sqrt(200), floor'
        },
        {
            'key': 'cargo_space_taken',
            'dim': 9,
            'op': partial(clip_one_hot, num=9),
            'other': 'one-hot'
        },
        {
            'key': 'cargo_space_max',
            'dim': 9,
            'op': partial(clip_one_hot, num=9),
            'other': 'one-hot'
        },  # 1020 wormhole
        {
            'key': 'display_type',
            'dim': 5,
            'op': partial(one_hot, num=5),
            'other': 'one-hot'
        },
        {
            'key': 'x',
            'dim': LOCATION_BIT_NUM,
            'op': partial(batch_binary_encode, bit_num=LOCATION_BIT_NUM),
            'other': 'binary encoding'
        },
        {
            'key': 'y',
            'dim': LOCATION_BIT_NUM,
            'op': partial(batch_binary_encode, bit_num=LOCATION_BIT_NUM),
            'other': 'binary encoding'
        },
        {
            'key': 'cloak',
            'dim': 5,
            'op': partial(one_hot, num=5),
            'other': 'one-hot'
        },
        {
            'key': 'is_powered',
            'dim': 2,
            'op': partial(one_hot, num=2),
            'other': 'one-hot'
        },
        {
            'key': 'hallucination',
            'dim': 2,
            'op': partial(one_hot, num=2),
            'other': 'one-hot'
        },
        {
            'key': 'active',
            'dim': 2,
            'op': partial(one_hot, num=2),
            'other': 'one-hot'
        },
        {
            'key': 'is_on_screen',
            'dim': 2,
            'op': partial(one_hot, num=2),
            'other': 'one-hot'
        },
        {
            'key': 'is_in_cargo',
            'dim': 2,
            'op': partial(one_hot, num=2),
            'other': 'one-hot'
        },
        {
            'key': 'mineral_contents',
            'dim': 20,
            'op': partial(div_one_hot, max_val=1900, ratio=100),
            'other': 'one-hot, 1900/100'
        },
        {
            'key': 'vespene_contents',
            'dim': 27,
            'op': partial(div_one_hot, max_val=2600, ratio=100),
            'other': 'one-hot, 2600/100'
        },
        {
            'key': 'assigned_harvesters',
            'dim': 35,
            'op': partial(clip_one_hot, num=35),
            'other': 'one-hot'
        },  # 34
        {
            'key': 'ideal_harvesters',
            'dim': 18,
            'op': partial(clip_one_hot, num=18),
            'other': 'one-hot'
        },  # 20
        {
            'key': 'weapon_cooldown',
            'dim': 32,
            'op': partial(clip_one_hot, num=32),
            'other': 'one-hot, game steps'
        },  # 35??
        {
            'key': 'order_length',
            'dim': 9,
            'op': partial(clip_one_hot, num=9),
            'other': 'one-hot'
        },
        {
            'key': 'order_id_0',
            'dim': NUM_ABILITIES,
            'op': partial(
                reorder_one_hot_array, array=ACTIONS_REORDER_ARRAY, num=NUM_ACTIONS, transform=ACT_TO_GENERAL_ACT_ARRAY
            ),
            'other': 'one-hot'
        },  # noqa
        {
            'key': 'order_id_1',
            'dim': NUM_ORDER_ACTIONS,
            'op': partial(
                reorder_one_hot_array,
                array=ORDER_ACTIONS_REORDER_ARRAY,
                num=NUM_ORDER_ACTIONS,
                transform=ACT_TO_GENERAL_ACT_ARRAY
            ),
            'other': 'one-hot'
        },  # TODO only building order  # noqa
        {
            'key': 'order_id_2',
            'dim': NUM_ORDER_ACTIONS,
            'op': partial(
                reorder_one_hot_array,
                array=ORDER_ACTIONS_REORDER_ARRAY,
                num=NUM_ORDER_ACTIONS,
                transform=ACT_TO_GENERAL_ACT_ARRAY
            ),
            'other': 'one-hot'
        },  # TODO only building order  # noqa
        {
            'key': 'order_id_3',
            'dim': NUM_ACTIONS,
            'op': partial(
                reorder_one_hot_array,
                array=ORDER_ACTIONS_REORDER_ARRAY,
                num=NUM_ORDER_ACTIONS,
                transform=ACT_TO_GENERAL_ACT_ARRAY
            ),
            'other': 'one-hot'
        },  # TODO only building order  # noqa
        {
            'key': 'buff_id_0',
            'dim': NUM_BUFFS,
            'op': partial(reorder_one_hot_array, array=BUFFS_REORDER_ARRAY, num=NUM_BUFFS),
            'other': 'one-hot'
        },
        {
            'key': 'buff_id_1',
            'dim': NUM_BUFFS,
            'op': partial(reorder_one_hot_array, array=BUFFS_REORDER_ARRAY, num=NUM_BUFFS),
            'other': 'one-hot'
        },
        {
            'key': 'addon_unit_type',
            'dim': NUM_ADDON,
            'op': partial(reorder_one_hot_array, array=ADDON_REORDER_ARRAY, num=NUM_ADDON),
            'other': 'one-hot'
        },
        {
            'key': 'order_progress_0',
            'dim': 10,
            'op': partial(div_one_hot, max_val=90, ratio=10),
            'other': 'one-hot(1/0.1)'
        },
        {
            'key': 'order_progress_1',
            'dim': 10,
            'op': partial(div_one_hot, max_val=90, ratio=10),
            'other': 'one-hot(1/0.1)'
        },
        {
            'key': 'attack_upgrade_level',
            'dim': 4,
            'op': partial(one_hot, num=4),
            'other': 'one-hot'
        },
        {
            'key': 'armor_upgrade_level',
            'dim': 4,
            'op': partial(one_hot, num=4),
            'other': 'one-hot'
        },
        {
            'key': 'shield_upgrade_level',
            'dim': 4,
            'op': partial(one_hot, num=4),
            'other': 'one-hot'
        },
        # {'key': 'was_selected', 'dim': 2, 'other': 'one-hot, last action'},
        # {'key': 'was_targeted', 'dim': 2, 'other': 'one-hot, last action'},
    ]
    return template


def transform_spatial_data():
    template = [
        # {'key': 'scattered_entities', 'other': '32 channel float'},
        {
            'key': 'height_map',
            'dim': 1,
            'op': partial(div_func, other=256., unsqueeze_dim=0),
            'other': 'float height_map/256'
        },
        {
            'key': 'camera',
            'dim': 2,
            'op': partial(num_first_one_hot, num=2),
            'other': 'one-hot 2 value'
        },
        {
            'key': 'visibility',
            'dim': 4,
            'op': partial(num_first_one_hot, num=4),
            'other': 'one-hot 4 value'
        },
        {
            'key': 'creep',
            'dim': 2,
            'op': partial(num_first_one_hot, num=2),
            'other': 'one-hot 2 value'
        },
        {
            'key': 'entity_owners',
            'dim': 5,
            'op': partial(num_first_one_hot, num=5),
            'other': 'one-hot 5 value'
        },
        {
            'key': 'alerts',
            'dim': 2,
            'op': partial(num_first_one_hot, num=2),
            'other': 'one-hot 2 value'
        },
        {
            'key': 'pathable',
            'dim': 2,
            'op': partial(num_first_one_hot, num=2),
            'other': 'one-hot 2 value'
        },
        {
            'key': 'buildable',
            'dim': 2,
            'op': partial(num_first_one_hot, num=2),
            'other': 'one-hot 2 value'
        },
        # {'key': 'effects', 'dim': 13, 'op': partial(num_first_one_hot, num=13), 'other': 'one-hot 13 value'},
    ]
    return template


def transform_scalar_data():
    template_obs = [
        {
            'key': 'agent_statistics',
            'arch': 'fc',
            'input_dim': 10,
            'ori': 'player',
            'output_dim': 64,
            'baseline_feature': True,
            'other': 'log(1+x)'
        },
        {
            'key': 'race',
            'arch': 'fc',
            'input_dim': 5,
            'output_dim': 32,
            'ori': 'home_race_requested',
            'op': partial(num_first_one_hot, num=5),
            'scalar_context': True,
            'other': 'one-hot 5 value'
        },
        {
            'key': 'enemy_race',
            'arch': 'fc',
            'input_dim': 5,
            'output_dim': 32,
            'ori': 'away_race_requested',
            'op': partial(num_first_one_hot, num=5),
            'scalar_context': True,
            'other': 'one-hot 5 value'
        },  # TODO 10% hidden  # noqa
        {
            'key': 'upgrades',
            'arch': 'fc',
            'input_dim': NUM_UPGRADES,
            'output_dim': 128,
            'ori': 'upgrades',
            'op': partial(reorder_boolean_vector, dictionary=UPGRADES_REORDER, num=NUM_UPGRADES),
            'baseline_feature': True,
            'other': 'boolean'
        },  # noqa
        {
            'key': 'enemy_upgrades',
            'arch': 'fc',
            'input_dim': 48,
            'output_dim': 128,
            'ori': 'enemy_upgrades',
            'other': 'boolean'
        },
        {
            'key': 'time',
            'arch': 'transformer position encode',
            'input_dim': 64,
            'output_dim': 64,
            'ori': 'game_loop',
            'op': get_postion_vector,
            'other': 'no further operation'
        },
        {
            'key': 'available_actions',
            'arch': 'fc',
            'input_dim': NUM_ACTIONS,
            'output_dim': 64,
            'ori': 'available_actions',
            'scalar_context': True,
            'other': 'boolean vector',
            #     'op': partial(
            #         reorder_boolean_vector, dictionary=ACTIONS_REORDER, num=NUM_ACTIONS, transform=ACT_TO_GENERAL_ACT
            #     )
            'op': lambda x: x,
        },
        {
            'key': 'unit_counts_bow',
            'arch': 'fc',
            'input_dim': NUM_UNIT_TYPES,
            'output_dim': 128,
            'ori': 'unit_counts',
            'baseline_feature': True,
            'other': 'square root'
        },
    ]
    template_replay = [
        {
            'key': 'mmr',
            'arch': 'fc',
            'input_dim': 7,
            'output_dim': 64,
            'op': partial(div_one_hot, max_val=6000, ratio=1000),
            'other': 'min(mmr / 1000, 6)'
        },
        {
            'key': 'cumulative_stat',
            'arch': 'multi_fc',
            'input_dims': OrderedDict(
                [
                    ('unit_build', NUM_UNIT_BUILD_ACTIONS), ('effect', NUM_EFFECT_ACTIONS),
                    ('research', NUM_RESEARCH_ACTIONS)
                ]
            ),
            'output_dim': 32,
            'scalar_context': True,
            'other': 'boolean vector, split and concat'
        },
        {
            'key': 'beginning_build_order',
            'arch': 'transformer',
            'input_dim': NUM_BEGIN_ACTIONS + LOCATION_BIT_NUM * 2,
            'output_dim': 32,
            'scalar_context': True,
            'baseline_feature': True,
            'other': 'transformer'
        },
    ]
    template_action = [
        {
            'key': 'last_delay',
            'arch': 'fc',
            'input_dim': 128,
            'output_dim': 64,
            'ori': 'action',
            'op': partial(clip_one_hot, num=128),
            'other': 'int'
        },
        {
            'key': 'last_queued',
            'arch': 'fc',
            'input_dim': 10,
            'output_dim': 256,
            'ori': 'action',
            'op': partial(num_first_one_hot, num=3),
            'other': 'one-hot 3'
        },  # 0 False 1 True 2 None
        {
            'key': 'last_action_type',
            'arch': 'fc',
            'input_dim': NUM_ACTIONS,
            'output_dim': 128,
            'ori': 'action',
            'op': partial(reorder_one_hot_array, array=ACTIONS_REORDER_ARRAY, num=NUM_ACTIONS),
            'other': 'one-hot NUM_ACTIONS'
        },  # noqa
    ]
    return template_obs, template_replay, template_action


def compress_obs(obs):
    # TODO: produce compressed obs directly without one_hot encoding, expecting a ~15% performance improvement
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_no_bool = 4
    new_obs['entity_info'] = {}
    new_obs['entity_info']['no_bool'] = obs['entity_info'][:, :entity_no_bool].numpy()
    entity_bool = obs['entity_info'][:, entity_no_bool:].to(torch.uint8).numpy()
    new_obs['entity_info']['bool_ori_shape'] = entity_bool.shape
    B, N = entity_bool.shape
    N_strided = N if N % 8 == 0 else (N // 8 + 1) * 8
    new_obs['entity_info']['bool_strided_shape'] = (B, N_strided)
    if N != N_strided:
        entity_bool = np.concatenate([entity_bool, np.zeros((B, N_strided - N), dtype=np.uint8)], axis=1)
    new_obs['entity_info']['bool'] = np.packbits(entity_bool)

    spatial_no_bool = 1
    new_obs['spatial_info'] = {}
    spatial_bool = obs['spatial_info'][spatial_no_bool:].to(torch.uint8).numpy()
    spatial_uint8 = obs['spatial_info'][:spatial_no_bool].mul_(256).to(torch.uint8).numpy()
    new_obs['spatial_info']['no_bool'] = spatial_uint8
    new_obs['spatial_info']['bool_ori_shape'] = spatial_bool.shape
    new_obs['spatial_info']['bool'] = np.packbits(spatial_bool)
    return new_obs


def decompress_obs(obs):
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_bool = np.unpackbits(obs['entity_info']['bool']).reshape(*obs['entity_info']['bool_strided_shape'])
    if obs['entity_info']['bool_strided_shape'][1] != obs['entity_info']['bool_ori_shape'][1]:
        entity_bool = entity_bool[:, :obs['entity_info']['bool_ori_shape'][1]]
    entity_no_bool = obs['entity_info']['no_bool']
    spatial_bool = np.unpackbits(obs['spatial_info']['bool']).reshape(*obs['spatial_info']['bool_ori_shape'])
    spatial_uint8 = obs['spatial_info']['no_bool'].astype(np.float32) / 256.
    new_obs['entity_info'] = torch.cat([torch.FloatTensor(entity_no_bool), torch.FloatTensor(entity_bool)], dim=1)
    new_obs['spatial_info'] = torch.cat([torch.FloatTensor(spatial_uint8), torch.FloatTensor(spatial_bool)], dim=0)
    return new_obs


def transform_cum_stat(cumulative_stat):
    print('transform_cum_stat moved to envs/statistics.py, should not be used')
    cumulative_stat_tensor = {
        'unit_build': torch.zeros(NUM_UNIT_BUILD_ACTIONS),
        'effect': torch.zeros(NUM_EFFECT_ACTIONS),
        'research': torch.zeros(NUM_RESEARCH_ACTIONS)
    }
    for k, v in cumulative_stat.items():
        if v['goal'] in ['unit', 'build']:
            cumulative_stat_tensor['unit_build'][UNIT_BUILD_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['effect']:
            cumulative_stat_tensor['effect'][EFFECT_ACTIONS_REORDER_ARRAY[k]] = 1
        elif v['goal'] in ['research']:
            cumulative_stat_tensor['research'][RESEARCH_ACTIONS_REORDER_ARRAY[k]] = 1
    return cumulative_stat_tensor


def transform_stat(stat, meta, location_num=LOCATION_BIT_NUM):
    print('transform_stat moved to envs/statistics.py, should not be used')
    beginning_build_order = stat['begin_statistics']
    beginning_build_order_tensor = []
    for item in beginning_build_order:
        action_type, location = item['action_type'], item['location']
        action_type = torch.LongTensor([action_type])
        action_type = reorder_one_hot_array(action_type, BEGIN_ACTIONS_REORDER_ARRAY, num=NUM_BEGIN_ACTIONS)
        if location == 'none':
            location = torch.zeros(location_num * 2)
        else:
            x = batch_binary_encode(torch.LongTensor([location[0]]), bit_num=location_num)[0]
            y = batch_binary_encode(torch.LongTensor([location[1]]), bit_num=location_num)[0]
            location = torch.cat([x, y], dim=0)
        beginning_build_order_tensor.append(torch.cat([action_type.squeeze(0), location], dim=0))
    beginning_build_order_tensor = torch.stack(beginning_build_order_tensor, dim=0)
    cumulative_stat_tensor = transform_cum_stat(stat['cumulative_statistics'])
    mmr = meta['home_mmr']
    mmr = torch.LongTensor([mmr])
    mmr = div_one_hot(mmr, 6000, 1000).squeeze(0)
    return {
        'mmr': mmr,
        'beginning_build_order': beginning_build_order_tensor,
        'cumulative_stat': cumulative_stat_tensor
    }  # noqa
