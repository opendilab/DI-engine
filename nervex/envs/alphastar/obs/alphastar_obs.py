'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. parse numpy arrays observations into tensors that pytorch can use
    2. compress and decompress the processed data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import torch
from functools import partial
from typing import Optional

from pysc2.lib.features import FeatureUnit
from pysc2.lib.action_dict import ACT_TO_GENERAL_ACT, ACT_TO_GENERAL_ACT_ARRAY
from pysc2.lib.static_data import NUM_BUFFS, NUM_ABILITIES, NUM_UNIT_TYPES, UNIT_TYPES_REORDER,\
     UNIT_TYPES_REORDER_ARRAY, BUFFS_REORDER_ARRAY, ABILITIES_REORDER_ARRAY, NUM_UPGRADES, UPGRADES_REORDER,\
     UPGRADES_REORDER_ARRAY, NUM_ACTIONS, ACTIONS_REORDER_ARRAY, ACTIONS_REORDER, NUM_ADDON, ADDON_REORDER_ARRAY,\
     NUM_BEGIN_ACTIONS, NUM_UNIT_BUILD_ACTIONS, NUM_EFFECT_ACTIONS, NUM_RESEARCH_ACTIONS,\
     UNIT_BUILD_ACTIONS_REORDER_ARRAY, EFFECT_ACTIONS_REORDER_ARRAY, RESEARCH_ACTIONS_REORDER_ARRAY,\
     BEGIN_ACTIONS_REORDER_ARRAY, NUM_ORDER_ACTIONS, ORDER_ACTIONS_REORDER_ARRAY
from collections import OrderedDict
from nervex.torch_utils import one_hot
from nervex.envs.common import EnvElement, num_first_one_hot, sqrt_one_hot, div_one_hot,\
    reorder_one_hot_array, div_func, batch_binary_encode, reorder_boolean_vector, clip_one_hot,\
    get_postion_vector
from ..action.alphastar_available_actions import get_available_actions_raw_data
from .alphastar_enemy_upgrades import get_enemy_upgrades_raw_data

LOCATION_BIT_NUM = 10
DELAY_BIT_NUM = 6
ENTITY_INFO_DIM = 1340


class SpatialObs(EnvElement):
    '''
        Overview: parse spatial observation into tensors
    '''
    _name = "AlphaStarSpatialObs"

    # override
    def _init(self, cfg: dict) -> None:
        '''
            Overview: initial related attributes
            Arguments:
                - cfg (:obj:'list'): wrapper config
        '''
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
        self.template = [
            {
                'key': 'height_map',
                'dim': 1,
                'op': partial(div_func, other=256., unsqueeze_dim=0),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float height_map/256'
            },
            {
                'key': 'camera',
                'dim': 2,
                'op': partial(num_first_one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 2 value'
            },
            {
                'key': 'visibility',
                'dim': 4,
                'op': partial(num_first_one_hot, num=4),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 4 value'
            },
            {
                'key': 'creep',
                'dim': 2,
                'op': partial(num_first_one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 2 value'
            },
            {
                'key': 'entity_owners',
                'dim': 5,
                'op': partial(num_first_one_hot, num=5),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 5 value'
            },
            {
                'key': 'alerts',
                'dim': 2,
                'op': partial(num_first_one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 2 value'
            },
            {
                'key': 'pathable',
                'dim': 2,
                'op': partial(num_first_one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 2 value'
            },
            {
                'key': 'buildable',
                'dim': 2,
                'op': partial(num_first_one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 2 value',
            },
        ]
        self.cfg = cfg
        self.spatial_resolution = cfg.spatial_resolution
        self._shape = tuple([self.channel_dim, *self.spatial_resolution])
        self._value = {'min': 0, 'max': 1, 'dtype': float, 'dinfo': 'float(0) + one_hot(1:)'}
        self._to_agent_processor = self.parse
        self._from_agent_processor = None

    # override
    def _details(self) -> str:
        return '3-dim [CxHxW] spatial observation'

    def _parse(self, feature: np.ndarray, idx_dict: dict) -> list:
        '''
            Overview: find corresponding setting in cfg, parse the feature
            Arguments:
                - feature (:obj:`ndarray`): the feature to parse
                - idx_dict (:obj:`dict`): feature index dict
            Returns:
                - ret (:obj:`list`): parse result tensor list
        '''
        ret = []
        for item in self.template:
            key = item['key']
            if key in idx_dict.keys():
                idx = idx_dict[key]
                data = feature[idx]
                data = torch.LongTensor(data)
                data = item['op'](data)
                ret.append(data)
        return ret

    def parse(self, obs: dict) -> torch.Tensor:
        '''
            Overview: gather parse results from different feature, concatenate them
            Arguments:
                - obs (:obj:`dict`): observation dict
            Returns:
                - (:obj'LongTensor'): feature tensor
        '''
        ret = []
        feature_minimap = obs['feature_minimap']
        ret.extend(self._parse(feature_minimap, self.feature_minimap_id))

        return torch.cat(ret, dim=0)

    @property
    def channel_dim(self) -> int:
        return sum([t['dim'] for t in self.template])


class EntityObs(EnvElement):
    _name = "AlphaStarEntityObs"

    # override
    def _init(self, cfg: dict) -> None:
        self.template = [
            {
                'key': 'build_progress',
                'dim': 1,
                'op': partial(div_func, other=100.),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float [0, 1]'
            },
            {
                'key': 'health_ratio',
                'dim': 1,
                'op': partial(div_func, other=255.),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float [0, 1]'
            },
            {
                'key': 'shield_ratio',
                'dim': 1,
                'op': partial(div_func, other=255.),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float [0, 1]'
            },
            {
                'key': 'energy_ratio',
                'dim': 1,
                'op': partial(div_func, other=255.),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'float'
                },
                'other': 'float [0, 1]'
            },
            {
                'key': 'unit_type',
                'dim': NUM_UNIT_TYPES,
                'op': partial(reorder_one_hot_array, array=UNIT_TYPES_REORDER_ARRAY, num=NUM_UNIT_TYPES),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'alliance',
                'dim': 5,
                'op': partial(one_hot, num=5),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'health',
                'dim': 39,
                'op': partial(sqrt_one_hot, max_val=1500),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, sqrt(1500), floor'
            },
            {
                'key': 'shield',
                'dim': 32,
                'op': partial(sqrt_one_hot, max_val=1000),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, sqrt(1000), floor'
            },
            {
                'key': 'energy',
                'dim': 15,
                'op': partial(sqrt_one_hot, max_val=200),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, sqrt(200), floor'
            },
            {
                'key': 'cargo_space_taken',
                'dim': 9,
                'op': partial(clip_one_hot, num=9),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'cargo_space_max',
                'dim': 9,
                'op': partial(clip_one_hot, num=9),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },  # Zerg Wormhole can also be regarded as cargo with more than 1000 units
            {
                'key': 'display_type',
                'dim': 5,
                'op': partial(one_hot, num=5),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'x',
                'dim': LOCATION_BIT_NUM,
                'op': partial(batch_binary_encode, bit_num=LOCATION_BIT_NUM),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'binary_encode'
                },
                'other': 'binary encoding'
            },
            {
                'key': 'y',
                'dim': LOCATION_BIT_NUM,
                'op': partial(batch_binary_encode, bit_num=LOCATION_BIT_NUM),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'binary_encode'
                },
                'other': 'binary encoding'
            },
            {
                'key': 'cloak',
                'dim': 5,
                'op': partial(one_hot, num=5),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'is_powered',
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'hallucination',  # such as hallucination unit produced by Protoss Sentry
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'active',  # such as producing units or research
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'is_on_screen',
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'is_in_cargo',
                'dim': 2,
                'op': partial(one_hot, num=2),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'mineral_contents',
                'dim': 20,
                'op': partial(div_one_hot, max_val=1900, ratio=100),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, 1900/100'
            },
            {
                'key': 'vespene_contents',
                'dim': 27,
                'op': partial(div_one_hot, max_val=2600, ratio=100),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, 2600/100'
            },
            {
                'key': 'assigned_harvesters',
                'dim': 35,
                'op': partial(clip_one_hot, num=35),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'ideal_harvesters',
                'dim': 18,
                'op': partial(clip_one_hot, num=18),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'weapon_cooldown',
                'dim': 32,
                'op': partial(clip_one_hot, num=32),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, game steps'
            },
            {
                'key': 'order_length',
                'dim': 9,
                'op': partial(clip_one_hot, num=9),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'order_id_0',
                'dim': NUM_ACTIONS,
                'op': partial(
                    reorder_one_hot_array,
                    array=ACTIONS_REORDER_ARRAY,
                    num=NUM_ACTIONS,
                    transform=ACT_TO_GENERAL_ACT_ARRAY
                ),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'order_id_1',
                'dim': NUM_ORDER_ACTIONS,
                'op': partial(
                    reorder_one_hot_array,
                    array=ORDER_ACTIONS_REORDER_ARRAY,
                    num=NUM_ORDER_ACTIONS,
                    transform=ACT_TO_GENERAL_ACT_ARRAY
                ),
                'other': 'one-hot',
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
            },  # only order action(a subset of action)
            {
                'key': 'order_id_2',
                'dim': NUM_ORDER_ACTIONS,
                'op': partial(
                    reorder_one_hot_array,
                    array=ORDER_ACTIONS_REORDER_ARRAY,
                    num=NUM_ORDER_ACTIONS,
                    transform=ACT_TO_GENERAL_ACT_ARRAY
                ),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },  # only order action(a subset of action)
            {
                'key': 'order_id_3',
                'dim': NUM_ORDER_ACTIONS,
                'op': partial(
                    reorder_one_hot_array,
                    array=ORDER_ACTIONS_REORDER_ARRAY,
                    num=NUM_ORDER_ACTIONS,
                    transform=ACT_TO_GENERAL_ACT_ARRAY
                ),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },  # only order action(a subset of action)
            {
                'key': 'buff_id_0',
                'dim': NUM_BUFFS,
                'op': partial(reorder_one_hot_array, array=BUFFS_REORDER_ARRAY, num=NUM_BUFFS),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'buff_id_1',
                'dim': NUM_BUFFS,
                'op': partial(reorder_one_hot_array, array=BUFFS_REORDER_ARRAY, num=NUM_BUFFS),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'addon_unit_type',  # only Terran has this attribute
                'dim': NUM_ADDON,
                'op': partial(reorder_one_hot_array, array=ADDON_REORDER_ARRAY, num=NUM_ADDON),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'order_progress_0',
                'dim': 10,
                'op': partial(div_one_hot, max_val=90, ratio=10),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, 0-9'
            },
            {
                'key': 'order_progress_1',
                'dim': 10,
                'op': partial(div_one_hot, max_val=90, ratio=10),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, 0-9'
            },
            {
                'key': 'attack_upgrade_level',
                'dim': 4,
                'op': partial(clip_one_hot, num=4),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'armor_upgrade_level',  # Zerg Ultralisk can be up to armor_upgrade_level 4
                'dim': 4,
                'op': partial(clip_one_hot, num=4),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'shield_upgrade_level',
                'dim': 4,
                'op': partial(clip_one_hot, num=4),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot'
            },
            {
                'key': 'was_selected',
                'dim': 2,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, last action'
            },
            {
                'key': 'was_targeted',
                'dim': 2,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot, last action'
            },
        ]
        self.cfg = cfg
        self.use_raw_units = cfg.use_raw_units
        self.key = 'feature_units' if not self.use_raw_units else 'raw_units'

        # entity_num can be different from game frames
        entity_num = 314  # placeholder
        self.entity_attribute_dim = sum(item['dim'] for item in self.template)
        self._shape = tuple([entity_num, self.entity_attribute_dim])
        self._value = {'min': 0, 'max': 1, 'dtype': float, 'dinfo': 'float(:4) + one_hot(4:)'}
        self._to_agent_processor = self.parse
        self._from_agent_processor = None

    def parse(self, obs: dict) -> torch.Tensor:
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
        for idx, item in enumerate(self.template):
            key = item['key']
            if 'was_' in key:  # `was_` attribute is assigned in the following statement
                continue
            key_index = FeatureUnit[key]
            item_data = feature_unit[:, key_index]
            item_data = torch.LongTensor(item_data)
            try:
                item_data = item['op'](item_data)
            except KeyError as e:
                print('{} parse error, current key: {}'.format(self._name, key))
                raise KeyError
            ret.append(item_data)
        ret = list(zip(*ret))
        ret = [torch.cat(item, dim=0) for item in ret]
        ret = torch.stack(ret, dim=0)
        # `was_` attribute
        ret = self._get_last_action_entity_info(ret, entity_raw, obs['last_action'])
        assert ret.shape[-1] == self.entity_attribute_dim, '{}/{}'.format(ret.shape, self.entity_attribute_dim)
        return ret, entity_raw

    def _get_last_action_entity_info(self, obs: torch.Tensor, entity_raw: dict, last_action: dict) -> torch.Tensor:
        N = obs.shape[0]
        selected_units = last_action['selected_units']
        target_units = last_action['target_units']
        obs = torch.cat([obs, torch.empty(N, 4)], dim=1)

        selected_units = selected_units if isinstance(selected_units, list) else []
        obs[:, -3] = 0
        obs[:, -4] = 1
        ids_tensor = np.array(entity_raw['id'])
        for v in selected_units:
            selected = (ids_tensor == v)
            obs[selected, -3] = 1
            obs[selected, -4] = 0

        target_units = target_units if isinstance(target_units, list) else []
        obs[:, -1] = 0
        obs[:, -2] = 1
        for v in target_units:
            targeted = (ids_tensor == v)
            obs[targeted, -1] = 1
            obs[targeted, -2] = 0
        return obs

    # override
    def _details(self) -> str:
        return '2-dim [MxN] entity observation(M->entity num, N->entity attributes dim)'


class ScalarObs(EnvElement):
    _name = "AlphaStarScalarObs"

    # override
    def _init(self, cfg: dict) -> None:
        def tensor_wrapper(fn):
            def wrapper(data):
                data = torch.LongTensor(data)
                data = fn(data)
                data = data.squeeze()
                return data

            return wrapper

        self.tensor_wrapper = tensor_wrapper
        self.begin_num = cfg.begin_num

        self.template = [
            {
                'key': 'agent_statistics',
                'dim': 10,
                'ori': 'player',
                'op': self._parse_agent_statistics,
                'value': {
                    'min': 0,
                    'max': 'inf',
                    'dtype': float,
                    'dinfo': 'float'
                },
                'baseline_feature': True,
                'other': 'log(1+x)'
            },
            {
                'key': 'race',
                'dim': 5,
                'ori': 'home_race_requested',
                'op': tensor_wrapper(partial(num_first_one_hot, num=5)),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'scalar_context': True,
                'other': 'one-hot 5 value'
            },
            {
                'key': 'enemy_race',
                'dim': 5,
                'ori': 'away_race_requested',
                'op': tensor_wrapper(partial(num_first_one_hot, num=5)),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'scalar_context': True,
                'other': 'one-hot 5 value'
            },
            {
                'key': 'upgrades',
                'dim': NUM_UPGRADES,
                'ori': 'upgrades',
                'op': tensor_wrapper(partial(reorder_boolean_vector, dictionary=UPGRADES_REORDER, num=NUM_UPGRADES)),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'boolean vector'
                },
                'baseline_feature': True,
                'other': 'boolean'
            },
            {
                'key': 'enemy_upgrades',
                'dim': 48,
                'ori': 'raw_units',
                'op': self._parse_enemy_upgrades,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'boolean vector'
                },
                'other': 'boolean'
            },
            {
                'key': 'time',
                'dim': 64,
                'ori': 'game_loop',
                'op': get_postion_vector,
                'value': {
                    'min': -1,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'sin, cos'
                },
                'other': 'transformer position encode'
            },
            {
                'key': 'available_actions',
                'dim': NUM_ACTIONS,
                'op': get_available_actions_raw_data,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'boolean vector'
                },
                'scalar_context': True,
                'other': 'boolean vector',
            },
            {
                'key': 'unit_counts_bow',
                'dim': NUM_UNIT_TYPES,
                'ori': 'unit_counts',
                'op': self._parse_unit_counts,
                'value': {
                    'min': 0,
                    'max': 'inf',
                    'dtype': float,
                    'dinfo': 'count vector'
                },
                'baseline_feature': True,
                'other': 'square root'
            },
            {
                'key': 'mmr',
                'dim': 7,
                'ori': 'mmr',
                'op': lambda x: x,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'min(mmr / 1000, 6)'
            },
            {
                'key': 'cumulative_stat',
                'dim': OrderedDict(
                    [
                        ('unit_build', NUM_UNIT_BUILD_ACTIONS), ('effect', NUM_EFFECT_ACTIONS),
                        ('research', NUM_RESEARCH_ACTIONS)
                    ]
                ),
                'ori': 'cumulative_stat',
                'op': lambda x: x,
                'value': {
                    k: {
                        'min': 0,
                        'max': 1,
                        'dtype': float,
                        'dinfo': 'boolean vector'
                    }
                    for k in ['unit_build', 'effect', 'research']
                },
                'scalar_context': True,
                'other': 'boolean vector, split and concat'
            },
            {
                'key': 'beginning_build_order',
                'dim': NUM_BEGIN_ACTIONS + LOCATION_BIT_NUM * 2 + self.begin_num,
                'ori': 'beginning_build_order',
                'op': lambda x: x,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot + binary_encode'
                },
                'scalar_context': True,
                'baseline_feature': True,
                'other': 'transformer'
            },
            {
                'key': 'last_delay',
                'dim': 128,
                'ori': 'last_delay',
                'op': tensor_wrapper(partial(clip_one_hot, num=128)),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'int'
            },
            {
                'key': 'last_queued',
                'dim': 20,
                'ori': 'last_action',
                'op': self._parse_last_queued_repeat,
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot 20, 3 for queue, 17 for repeat'
            },  # 0 False 1 True 2 None
            {
                'key': 'last_action_type',
                'dim': NUM_ACTIONS,
                'ori': 'last_action_type',
                'op': tensor_wrapper(partial(reorder_one_hot_array, array=ACTIONS_REORDER_ARRAY, num=NUM_ACTIONS)),
                'value': {
                    'min': 0,
                    'max': 1,
                    'dtype': float,
                    'dinfo': 'one_hot'
                },
                'other': 'one-hot NUM_ACTIONS'
            },
        ]
        if cfg.use_score_cumulative:
            self.template += [
                {
                    'key': 'score_cumulative',
                    'dim': 13,
                    'ori': 'score_cumulative',
                    'op': lambda x: torch.log(1 + torch.FloatTensor(x)),
                    'value': {
                        'min': 0,
                        'max': 'inf',
                        'dtype': float,
                        'dinfo': 'float'
                    },
                    'other': 'log(1+x)'
                }
            ]
        self.cfg = cfg
        self._shape = {t['key']: t['dim'] for t in self.template}
        self._value = {t['key']: t['value'] for t in self.template}
        self._to_agent_processor = self.parse
        self._from_agent_processor = None

    def _parse_agent_statistics(self, data: np.ndarray) -> torch.Tensor:
        data = torch.FloatTensor(data)
        data = data[1:]
        return torch.log(data + 1)

    def _parse_unit_counts(self, data: dict, max_val: Optional[int] = 225) -> torch.Tensor:
        ret = torch.zeros(NUM_UNIT_TYPES)
        key = data.keys()
        val = list(data.values())
        val = np.sqrt(np.clip(np.array(val), 0, max_val))
        for k, v in zip(key, val):
            idx = UNIT_TYPES_REORDER[k]
            ret[idx] = v
        return ret

    def _parse_last_queued_repeat(self, last_action: dict) -> torch.Tensor:
        last_queued = last_action['queued']
        last_queued = last_queued if isinstance(last_queued, torch.Tensor) else torch.LongTensor([2])  # 2 as 'none'
        last_queued = self.tensor_wrapper(partial(num_first_one_hot, num=3))(last_queued)

        last_repeat = self.tensor_wrapper(partial(clip_one_hot, num=17))([last_action['repeat_count']])
        return torch.cat([last_queued, last_repeat], dim=0)

    def _parse_enemy_upgrades(self, raw_units: dict) -> torch.Tensor:
        if not hasattr(self, 'enemy_upgrades'):
            self.enemy_upgrades = None
        self.enemy_upgrades = get_enemy_upgrades_raw_data(raw_units, self.enemy_upgrades)
        return copy.deepcopy(self.enemy_upgrades)

    def parse(self, obs: dict) -> dict:
        obs['last_action_type'] = [obs['last_action']['action_type']]
        obs['last_delay'] = [obs['last_action']['delay']]  # one_hot need 1-dim

        ret = {}
        for idx, item in enumerate(self.template):
            key = item['key']
            if 'ori' in item:
                item_data = obs[item['ori']]
            else:
                item_data = obs
            item_data = item['op'](item_data)
            ret[key] = item_data
        return ret

    # override
    def _details(self) -> str:
        return 'dict including global scalar observation: {}'.format('\t'.join([t['key'] for t in self.template]))


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
