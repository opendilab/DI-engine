import copy
import enum
from collections import namedtuple
from functools import partial

import numpy as np
import torch
from pysc2.lib import actions
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK, ACT_TO_GENERAL_ACT
from pysc2.lib.static_data import NUM_ACTIONS, ACTIONS_REORDER, ACTIONS_REORDER_INV

from nervex.envs.common import EnvElement
from nervex.torch_utils import to_tensor, tensor_to_list

DELAY_MAX = 128


def action_unit_id_transform(data, inv=False):
    '''
    Overview: transform original game unit id in action to the current frame unit id
    '''

    def transform(frame):
        frame = copy.deepcopy(frame)
        id_list = frame['entity_raw']['id']
        if 'selected_units' in data.keys():
            units = frame['selected_units']
        elif 'target_units' in data.keys():
            units = frame['target_units']
        else:
            raise KeyError("invalid key in action_unit_id_transform frame: {}".format(frame.keys()))
        if units is not None:
            if np.isscalar(units):
                units = [units]
            unit_ids = []
            for unit in units:
                if inv:
                    unit_ids.append(id_list[unit])
                else:
                    if unit in id_list:
                        unit_ids.append(id_list.index(unit))
                    else:
                        raise Exception("not found id({}) in nearest observation".format(unit))
            return unit_ids
        else:
            return units

    if isinstance(data, list):
        for idx, item in enumerate(data):
            data[idx] = transform(item)
        return data
    elif isinstance(data, dict):
        return transform(data)
    else:
        raise TypeError("invalid input type: {}".format(type(data)))


def location_transform(data, inv):
    """
        Note:
            1. env location(x major), agent location(y major)
            2. if inv=True, agent->env; otherwise, env->agent
    """
    location, map_size = data['target_location'], data['map_size']
    if location is None:
        return location

    def location_check(x, y):
        try:
            assert x < map_size[0] and y < map_size[1], 'target location out of range, corrupt replay'
        except AssertionError:
            x = min(x, map_size[0] - 1)
            y = min(y, map_size[1] - 1)
            print('[WARNING]: abnormal location-map_size: {}/{}'.format(location, map_size))
        return [x, y]

    if inv:
        y, x = location
        y += 0.5  # building fix on .5 coordination
        x += 0.5
        y = map_size[1] - y
        x, y = location_check(x, y)
        location = [x, y]
    else:
        x, y = location
        y = map_size[1] - y
        x, y = int(x), int(y)
        x, y = location_check(x, y)
        location = [y, x]
    return location


class AlphaStarRawAction(EnvElement):
    _name = "AlphaStarRawAction"
    _action_keys = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
    Action = namedtuple('Action', _action_keys)

    def _init(self, cfg):
        self._map_size = cfg.map_size
        self._default_val = None
        self._selected_units_num = 32  # placeholder, [1,inf)
        self._template = {
            'action_type':
                {
                    'name': 'action_type',
                    'shape': (1,),
                    # value is a range[min, max)
                    'value': {  # action used by agent
                        'min': 0,
                        'max': NUM_ACTIONS,
                        'dtype': int,
                        'dinfo': 'int value',
                    },
                    'env_value': 'categorial value, refer to pysc2.lib.action_dict',
                    'to_agent_processor': lambda x: ACTIONS_REORDER[x],
                    'from_agent_processor': lambda x: ACTIONS_REORDER_INV[x],
                    'necessary': True,
                },
            'delay':
                {
                    'name': 'delay',
                    'shape': (1,),
                    'value': {  # action used by agent
                        'min': 0,
                        'max': DELAY_MAX,
                        'dtype': int,
                        'dinfo': 'int value',
                    },
                    'env_value': '[0, inf)',
                    'to_agent_processor': lambda x: min(x, DELAY_MAX),
                    'from_agent_processor': lambda x: x,
                    'necessary': True,
                },
            'queued':
                {
                    'name': 'queued',
                    'shape': (1,),
                    'value': {  # action used by agent
                        'min': 0,
                        'max': 2,
                        'dtype': int,
                        'dinfo': 'int value',
                    },
                    'env_value': 'bool',
                    'to_agent_processor': lambda x: x,
                    'from_agent_processor': lambda x: x,
                    'necessary': False,
                },
            'selected_units':
                {
                    'name': 'selected_units',
                    'shape': (self._selected_units_num,),
                    'value': {  # action used by agent
                        'min': 0,
                        'max': 'inf',
                        'dtype': int,
                        'dinfo': 'int value',
                    },
                    'env_value': 'unique entity id',
                    'to_agent_processor': partial(action_unit_id_transform, inv=False),
                    'from_agent_processor': partial(action_unit_id_transform, inv=True),
                    'necessary': False,
                    'other': 'value is entity index',
                },
            'target_units':
                {
                    'name': 'target_units',
                    'shape': (1,),
                    'value': {  # action used by agent
                        'min': 0,
                        'max': 'inf',
                        'dtype': int,
                        'dinfo': 'int value',
                    },
                    'env_value': 'unique entity id',
                    'to_agent_processor': partial(action_unit_id_transform, inv=False),
                    'from_agent_processor': partial(action_unit_id_transform, inv=True),
                    'necessary': False,
                    'other': 'value is entity index',
                },
            'target_location':
                {
                    'name': 'target_location',
                    'shape': (2,),
                    'value': {  # action used by agent
                        'min': (0, 0),
                        'max': self._map_size,
                        'dtype': float,
                        'dinfo': 'float value',
                    },
                    'env_value': 'float value',
                    'to_agent_processor': partial(location_transform, inv=False),
                    'from_agent_processor': partial(location_transform, inv=True),
                    'necessary': False,
                    'other': 'agent value use round env value',
                },
        }
        self._shape = {t['name']: t['shape'] for t in self._template.values()}
        self._value = {t['name']: t['value'] for t in self._template.values()}
        self._replay_action_helper = AlphaStarReplayActionHelper()

    def _get_output_template(self):
        template = {k: None for k in self._action_keys}
        return template

    def _content_processor(self, data, key):
        action = data['action']
        entity_raw = data['entity_raw']
        if 'map_size' in data.keys():
            map_size = data['map_size']
        else:
            map_size = self._map_size

        for k, v in action.items():
            if k in ['selected_units', 'target_units']:
                action_key = {k: v, 'entity_raw': entity_raw}
            elif k == 'target_location':
                action_key = {k: v, 'map_size': map_size}
            else:
                action_key = v
            action[k] = self._template[k][key](action_key)
        return action

    def _to_agent_processor(self, data):
        # replay action parse
        data['action'] = self._replay_action_helper(data['action'])
        # content processor
        action = self._content_processor(data, 'to_agent_processor')
        # format processor
        action = to_tensor(action)
        return action

    def _from_agent_processor(self, data):
        # format processor
        data['action'] = tensor_to_list(data['action'])
        # content processor
        action = self._content_processor(data, 'from_agent_processor')
        return AlphaStarRawAction.Action(**action)

    # override
    def _details(self):
        return '\t'.join(self._action_keys)


class AlphaStarReplayActionHelper:
    def __init__(self):
        self._action_keys = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self._action_template = {k: None for k in self._action_keys}
        self._ability2action = {
            'camera_move': self._parse_raw_camera_move,
            'unit_command': self._parse_raw_unit_command,
            'toggle_autocast': self._parse_raw_toggle_autocast,
        }

    def __call__(self, data):
        action = data
        ret = copy.deepcopy(self._action_template)
        for k, f in self._ability2action.items():
            act_val = getattr(action, k)
            v = f(act_val)
            if v is not None:
                ret[k] = v
        return list(ret.values())

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_camera_move(self, t):
        if t.HasField('center_world_space'):
            location = (t.center_world_space.x, t.center_world_space.y)
            return {'action_type': [168], 'target_location': location}  # raw_camera_move 168
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_unit_command(self, t):
        if t.HasField('ability_id'):
            try:
                ret = {'selected_units': list(t.unit_tags)}
                # target_units and target_location can't exist at the same time
                assert ((t.HasField('target_world_space_pos')) + (t.HasField('target_unit_tag')) <= 1)
                if t.HasField('target_world_space_pos'):
                    ret['target_location'] = (t.target_world_space_pos.x, t.target_world_space_pos.y)
                    ret['action_type'] = self._ability_to_raw_func(t.ability_id, actions.raw_cmd_pt)
                else:
                    if t.HasField('target_unit_tag'):
                        ret['target_units'] = [t.target_unit_tag]
                        ret['action_type'] = self._ability_to_raw_func(t.ability_id, actions.raw_cmd_unit)
                    else:
                        ret['action_type'] = self._ability_to_raw_func(t.ability_id, actions.raw_cmd)
                # transform into general_id action
                ret['action_type'] = [ACT_TO_GENERAL_ACT[ret['action_type']]]
                # queued attr
                has_queue_attr = GENERAL_ACTION_INFO_MASK[ret['action_type']]['queued']
                if has_queue_attr:
                    if t.HasField('queue_command'):
                        assert (t.queue_command)
                        ret['queued'] = [t.queue_command]
                    else:
                        ret['queued'] = [False]
                return ret
            except RuntimeError:
                return None
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_toggle_autocast(self, t):
        if t.HasField('ability_id'):
            try:
                ret = {'action_type': [self._ability_to_raw_func(t.ability_id, actions.raw_autocast)]}
                ret['selected_units'] = list(t.unit_tags)
                # transfrom into general_id action
                ret['action_type'] = [ACT_TO_GENERAL_ACT[ret['action_type'][0]]]
                return ret
            except RuntimeError:
                return None
        else:
            return None

    def _ability_to_raw_func(self, ability_id, cmd_type):
        if ability_id not in actions.RAW_ABILITY_IDS:
            print("unknown ability id: {}".format(ability_id))
            raise RuntimeError("unknown ability id: {}".format(ability_id))
        for func in actions.RAW_ABILITY_IDS[ability_id]:
            if func.function_type is cmd_type:
                return func.id
        print("not found corresponding cmd type, id: {}\tcmd type: {}".format(ability_id, cmd_type))
        raise RuntimeError("not found corresponding cmd type, id: {}\tcmd type: {}".format(ability_id, cmd_type))


def merge_same_id_action(actions):
    def merge(same_id_actions):
        def apply_merge(action_list):
            selected_units = []
            for a in action_list:
                selected_units.extend(a['selected_units'])
            selected_units = list(set(selected_units))  # remove repeat element
            action_list[0]['selected_units'] = selected_units
            return [action_list[0]]

        def merge_by_key(key):
            sames = {}
            for a in same_id_actions:
                k = a[key]
                if isinstance(k, list):
                    k = '-'.join([str(t) for t in k])
                if k not in sames.keys():
                    sames[k] = [a]
                else:
                    sames[k].append(a)
            ret = []
            for k, v in sames.items():
                a = apply_merge(v) if len(v) > 1 else v
                ret.extend(a)
            return ret

        action_type = same_id_actions[0]['action_type'][0]
        if action_type == 0:  # no op
            return [same_id_actions[0]]
        if same_id_actions[0]['target_units'] is not None:
            # target_units
            return merge_by_key('target_units')
        elif same_id_actions[0]['target_location'] is not None:
            # target_location
            return merge_by_key('target_location')
        else:
            return apply_merge(same_id_actions)

    same_action_dict = {}
    for a in actions:
        k = a['action_type'][0]
        if k not in same_action_dict.keys():
            same_action_dict[k] = [a]
        else:
            same_action_dict[k].append(a)
    ret = []
    for k, v in same_action_dict.items():
        if len(v) > 1:
            ret.extend(merge(v))
        else:
            ret.append(v[0])
    return to_tensor(ret, torch.long)


def remove_repeat_data(data, min_delay=16, max_move=3, target_action_type_list=[168, 12, 3]):
    '''
        168(camera move), 12(smart unit), 3(attack unit),
    '''

    class State(enum.IntEnum):
        init = 0,
        add = 1,

    def merge(selected_list):

        if len(selected_list) == 1:
            return selected_list

        def single_action_merge(start, end, check_delay=True):
            part = selected_list[start:end]
            if len(part) <= 1:
                return part
            actions = [p['actions'] for p in part]
            # high delay
            if check_delay:
                high_delay_step = [idx for idx, a in enumerate(actions) if a['delay'] >= min_delay]
                result = []
                cur = start
                for i in high_delay_step:
                    result.extend(single_action_merge(cur, start + i, False))
                    cur = start + i
                if cur < end:
                    result.extend(single_action_merge(cur, end, False))
            else:

                def equal(a, b):
                    if type(a) != type(b):
                        return False
                    if isinstance(a, torch.Tensor):
                        if a.shape != b.shape:
                            return False
                        return (a == b).all()
                    else:
                        return a == b

                # target units
                if isinstance(actions[0]['target_units'], torch.Tensor):
                    # same selected units and target_units
                    a0_s_units = actions[0]['selected_units']
                    not_same_s = [idx for idx, a in enumerate(actions) if not equal(a['selected_units'], a0_s_units)]
                    a0_t_units = actions[0]['target_units']
                    not_same_t = [idx for idx, a in enumerate(actions) if not equal(a['target_units'], a0_t_units)]
                    not_same = list(set(not_same_s).union(set(not_same_t)))
                    result = [part[0]]
                    if len(not_same) > 0:
                        print('not same selected_units and target_units\n', actions)
                        result.extend(single_action_merge(start + not_same[0], end, False))
                # target location
                else:
                    # same selected_units units
                    a0_s_units = actions[0]['selected_units']
                    not_same = [idx for idx, a in enumerate(actions) if not equal(a['selected_units'], a0_s_units)]
                    if len(not_same) > 0:
                        print('not same selected_units\n', actions, not_same)
                        result = []
                        result.extend(single_action_merge(start, start + not_same[0], False))
                        result.extend(single_action_merge(start + not_same[0] + 1, end, False))
                    else:
                        location = torch.stack([a['target_location'] for a in actions], dim=0).float()
                        x, y = torch.chunk(location, 2, dim=1)
                        x_flag = torch.abs(x - x.mean()).max() > max_move
                        y_flag = torch.abs(y - y.mean()).max() > max_move
                        if x_flag or y_flag:
                            result = [part[0], part[-1]]
                        else:
                            part[0]['actions']['target_location'] = torch.FloatTensor([x.mean(),
                                                                                       y.mean()]).round().long()  # noqa
                            result = [part[0]]
            return result

        start = 0
        start_action_type = selected_list[start]['actions']['action_type']
        result = []
        for idx in range(len(selected_list)):
            if start_action_type != selected_list[idx]['actions']['action_type']:
                result.extend(single_action_merge(start, idx))
                start = idx
                start_action_type = selected_list[start]['actions']['action_type']
        if start < len(selected_list):
            result.extend(single_action_merge(start, len(selected_list)))
        '''
        print('-'*60 + '\nnum:{}\n'.format(len(selected_list)))
        for item in selected_list:
            print(item['actions'])
        print('*'*60 + '\nnum:{}\n'.format(len(result)))
        for item in result:
            print(item['actions'])
        '''
        return result

    new_data = []
    state = State.init
    selected_list = []
    for step in data:
        action = step['actions']
        action_type = action['action_type']
        if state == State.init:
            if action_type in target_action_type_list:
                state = State.add
                assert (len(selected_list) == 0)
                selected_list.append(step)
            else:
                new_data.append(step)
        elif state == State.add:
            if action_type in target_action_type_list:
                selected_list.append(step)
            else:
                state = State.init
                new_data.extend(merge(selected_list))
                selected_list = []
                new_data.append(step)
    return new_data
