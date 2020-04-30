'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. parse numpy arrays actions into tensors for pytorch
'''

import collections
import torch
import enum
import copy
from pysc2.lib import actions
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK, ACT_TO_GENERAL_ACT
from sc2learner.torch_utils import to_tensor


class AlphastarActParser(object):
    '''
        Overview: parse action into tensors
        Interface: __init__, parse
    '''
    def __init__(self, feature_layer_resolution, map_size, use_resolution):
        '''
            Overview: initial related attributes
            Arguments:
                - feature_layer_resolution (:obj:'int'): feature layer resolution
                - map_size (:obj:'list'): map size (x, y format)
        '''
        self.input_template = {
            'camera_move': self._parse_raw_camera_move,
            'unit_command': self._parse_raw_unit_command,
            'toggle_autocast': self._parse_raw_toggle_autocast,
        }
        self.output_template = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self.map_size = map_size
        assert self.map_size[0] != 0 and self.map_size[1] != 0
        self.use_resolution = use_resolution
        self.resolution = feature_layer_resolution * 2

    def _get_output_template(self):
        template = {k: None for k in self.output_template}
        return template

    def parse(self, action):
        '''
            Overview: parse an action
            Arguments:
                - action (:obj:'ActionRaw'): raw action in proto format
            Returns:
                - ret (:obj:'list'): a list includes actions parsed from the raw action
        '''
        ret = {}
        for k, f in self.input_template.items():
            act_val = getattr(action, k)
            v = f(act_val)
            if v is not None:
                item = self._get_output_template()
                item.update(v)
                ret[k] = item
        return list(ret.values())

    def world_coord_to_minimap(self, coord):
        max_dim = max(self.map_size[0], self.map_size[1])   # spatial aspect ratio doesn't change with any resolution
        new_x = coord[0] * self.resolution / max_dim
        new_y = (self.map_size[1] - coord[1]) * self.resolution / max_dim
        return [new_y, new_x]   # spatial information is y major coordinate

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_camera_move(self, t):
        if t.HasField('center_world_space'):
            if self.use_resolution:
                location = self.world_coord_to_minimap((t.center_world_space.x, t.center_world_space.y))
            else:
                location = [self.map_size[1] - t.center_world_space.y, t.center_world_space.x]  # y major
            return {'action_type': [168], 'target_location': location}  # raw_camera_move 168
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_unit_command(self, t):
        if t.HasField('ability_id'):
            ret = {'selected_units': list(t.unit_tags)}
            # target_units and target_location can't exist at the same time
            assert ((t.HasField('target_world_space_pos')) + (t.HasField('target_unit_tag')) <= 1)
            if t.HasField('target_world_space_pos'):
                # origin world position
                if self.use_resolution:
                    ret['target_location'] = self.world_coord_to_minimap((t.target_world_space_pos.x,
                                                                          t.target_world_space_pos.y))
                else:
                    ret['target_location'] = [
                        self.map_size[1] - t.target_world_space_pos.y, t.target_world_space_pos.x
                    ]  # y major  # noqa
                ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd_pt)]
            else:
                if t.HasField('target_unit_tag'):
                    ret['target_units'] = [t.target_unit_tag]
                    ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd_unit)]
                else:
                    ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd)]
            if ret['action_type'] == [0]:
                ret = self._get_output_template()
                ret['action_type'] = [0]
            # transform into general_id action
            ret['action_type'] = [ACT_TO_GENERAL_ACT[ret['action_type'][0]]]
            # queued attr
            has_queue_attr = GENERAL_ACTION_INFO_MASK[ret['action_type'][0]]['queued']
            if has_queue_attr:
                if t.HasField('queue_command'):
                    assert (t.queue_command)
                    ret['queued'] = [t.queue_command]
                else:
                    ret['queued'] = [False]
            return ret
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_toggle_autocast(self, t):
        if t.HasField('ability_id'):
            ret = {'action_type': [self.ability_to_raw_func(t.ability_id, actions.raw_autocast)]}
            ret['selected_units'] = list(t.unit_tags)
            # transfrom into general_id action
            ret['action_type'] = [ACT_TO_GENERAL_ACT[ret['action_type'][0]]]
            return ret
        else:
            return None

    def ability_to_raw_func(self, ability_id, cmd_type):
        if ability_id not in actions.RAW_ABILITY_IDS:
            print("unknown ability id: {}".format(ability_id))
            return 0
        for func in actions.RAW_ABILITY_IDS[ability_id]:
            if func.function_type is cmd_type:
                return func.id
        print("not found corresponding cmd type, id: {}\tcmd type: {}".format(ability_id, cmd_type))
        return 0  # error case, regard as no op

    def merge_same_id_action(self, actions):
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


def action_unit_id_transform(data, inverse=False):
    '''
    Overview: transform original game unit id in action to the current frame unit id
    '''
    def transform(frame):
        frame = copy.deepcopy(frame)
        id_list = frame['entity_raw']['id']
        action = frame['actions']
        for k in ['selected_units', 'target_units']:
            if isinstance(action[k], torch.Tensor):
                unit_ids = []
                for unit in action[k]:
                    val = unit.item()
                    if inverse:
                        unit_ids.append(id_list[val])
                    else:
                        if val in id_list:
                            unit_ids.append(id_list.index(val))
                        else:
                            raise Exception("not found {} id({}) in nearest observation".format(k, val))
                frame['actions'][k] = torch.LongTensor(unit_ids)
        return frame

    if isinstance(data, list):
        for idx, item in enumerate(data):
            data[idx] = transform(item)
        return data
    elif isinstance(data, dict):
        return transform(data)
    else:
        raise TypeError("invalid input type: {}".format(type(data)))


class State(enum.IntEnum):
    init = 0,
    add = 1,


def remove_repeat_data(data, min_delay=16, max_move=3, target_action_type_list=[168, 12, 3]):
    '''
        168(camera move), 12(smart unit), 3(attack unit),
    '''
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


def test_merge_same_id_action():
    # fake data, the same format
    actions = [
        {
            'action_type': [0],
            'selected_units': None,
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [0],
            'selected_units': None,
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [112, 131],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [132],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [133],
            'target_units': [938],
            'target_location': None
        },
        {
            'action_type': [3],
            'selected_units': [132],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [4],
            'selected_units': [1321],
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [4],
            'selected_units': [1321, 1328],
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [5],
            'selected_units': [1321, 1328],
            'target_units': None,
            'target_location': [21, 43]
        },
        {
            'action_type': [5],
            'selected_units': [1322, 1327],
            'target_units': None,
            'target_location': [21, 43]
        },
        {
            'action_type': [5],
            'selected_units': [1323, 1326],
            'target_units': None,
            'target_location': [21, 42]
        },
    ]

    class Map:
        x = 128
        y = 128

    act_parser = AlphastarActParser(128, Map())
    merged_actions = act_parser.merge_same_id_action(actions)
    for k in merged_actions:
        print(k)


if __name__ == "__main__":
    test_merge_same_id_action()
