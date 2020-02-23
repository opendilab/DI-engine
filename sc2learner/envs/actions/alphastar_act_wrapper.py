'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. parse numpy arrays actions into tensors for pytorch
'''

import collections
import torch
from pysc2.lib import actions
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK, ACT_TO_GENERAL_ACT
from sc2learner.utils import to_tensor


class AlphastarActParser(object):
    '''
        Overview: parse action into tensors
        Interface: __init__, parse
    '''

    def __init__(self, feature_layer_resolution, map_size):
        '''
            Overview: initial related attributes
            Arguments:
                - feature_layer_resolution (:obj:'int'): feature layer resolution
                - map_size (:obj:'obj'): map size metadata in proto format
        '''
        self.input_template = {'camera_move': self._parse_raw_camera_move,
                               'unit_command': self._parse_raw_unit_command,
                               'toggle_autocast': self._parse_raw_toggle_autocast, }
        self.output_template = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self.map_size = (map_size.x, map_size.y)
        if isinstance(feature_layer_resolution, collections.Sequence):
            self.resolution = feature_layer_resolution
        else:
            self.resolution = (feature_layer_resolution, feature_layer_resolution)

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
        coord[0] = min(self.map_size[0], coord[0])
        coord[1] = min(self.map_size[1], coord[1])
        new_x = int(coord[0] * self.resolution[0] / (self.map_size[0] + 1e-3))
        new_y = int(coord[1] * self.resolution[1] / (self.map_size[1] + 1e-3))
        max_limit = self.resolution[0] * self.resolution[1]
        assert(new_x < max_limit and new_y < max_limit)
        return (new_x, new_y)

    def minimap_to_world_coord(self, location):
        assert(location[0] < self.resolution[0])
        assert(location[1] < self.resolution[1])
        new_x = location[0] * self.map_size[0] / self.resolution[0]
        new_y = location[1] * self.map_size[1] / self.resolution[1]
        return (new_x, new_y)

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_camera_move(self, t):
        if t.HasField('center_world_space'):
            location = [t.center_world_space.x, t.center_world_space.y]
            location = self.world_coord_to_minimap(location)
            return {'action_type': [168], 'target_location': location}  # raw_camera_move 168
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_unit_command(self, t):
        if t.HasField('ability_id'):
            ret = {'selected_units': t.unit_tags}
            # target_units and target_location can't exist at the same time
            assert((t.HasField('target_world_space_pos')) + (t.HasField('target_unit_tag')) <= 1)
            if t.HasField('target_world_space_pos'):
                # origin world position
                ret['target_location'] = [t.target_world_space_pos.x, t.target_world_space_pos.y]
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
            # transfrom into general_id action
            ret['action_type'] = [ACT_TO_GENERAL_ACT[ret['action_type'][0]]]
            # queued attr
            has_queue_attr = GENERAL_ACTION_INFO_MASK[ret['action_type'][0]]['queued']
            if has_queue_attr:
                if t.HasField('queue_command'):
                    assert(t.queue_command)
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
            if t.HasField('unit_tags'):
                ret['selected_units'] = t.unit_tags
            else:
                ret['action_type'] = [0]
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
            if same_id_actions[0]['target_units'][0] is not None:
                # target_units
                return merge_by_key('target_units')
            elif same_id_actions[0]['target_location'][0] is not None:
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


def test_merge_same_id_action():
    # fake data, the same format
    actions = [
        {'action_type': [0], 'selected_units': [None], 'target_units': [None], 'target_location': [None]},
        {'action_type': [0], 'selected_units': [None], 'target_units': [None], 'target_location': [None]},
        {'action_type': [2], 'selected_units': [112, 131], 'target_units': [939], 'target_location': [None]},
        {'action_type': [2], 'selected_units': [132], 'target_units': [939], 'target_location': [None]},
        {'action_type': [2], 'selected_units': [133], 'target_units': [938], 'target_location': [None]},
        {'action_type': [3], 'selected_units': [132], 'target_units': [939], 'target_location': [None]},
        {'action_type': [4], 'selected_units': [1321], 'target_units': [None], 'target_location': [None]},
        {'action_type': [4], 'selected_units': [1321, 1328], 'target_units': [None], 'target_location': [None]},
        {'action_type': [5], 'selected_units': [1321, 1328], 'target_units': [None], 'target_location': [21, 43]},
        {'action_type': [5], 'selected_units': [1322, 1327], 'target_units': [None], 'target_location': [21, 43]},
        {'action_type': [5], 'selected_units': [1323, 1326], 'target_units': [None], 'target_location': [21, 42]},
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
