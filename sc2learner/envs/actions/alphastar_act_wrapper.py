import torch
from pysc2.lib import actions


class AlphastarActParser(object):
    def __init__(self, feature_layer_resolution, map_size):
        self.input_template = {'camera_move': self._parse_raw_camera_move,
                               'unit_command': self._parse_raw_unit_command,
                               'toggle_autocast': self._parse_raw_toggle_autocast, }
        self.output_template = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']
        self.map_size = (map_size.x, map_size.y)
        if isinstance(feature_layer_resolution, tuple):
            self.resolution = feature_layer_resolution
        else:
            self.resolution = (feature_layer_resolution, feature_layer_resolution)

    def _get_output_template(self):
        template = {k: None for k in self.output_template}
        template['queued'] = [False]
        return template

    def parse(self, action):
        ret = {}
        for k, f in self.input_template.items():
            act_val = getattr(action, k)
            v = f(act_val)
            if v is not None:
                item = self._get_output_template()
                item.update(v)
                ret[k] = self.dict2tensor(item)
        return ret

    def world_coord_to_minimap(self, coord):
        new_x = round(coord[0] * self.resolution[0] / self.map_size[0])
        new_y = round(coord[1] * self.resolution[1] / self.map_size[1])
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
            if t.HasField('queue_command'):
                ret['queued'] = [t.queue_command]
            assert((t.HasField('target_world_space_pos')) + (t.HasField('target_unit_tag')) <= 1)
            if t.HasField('target_world_space_pos'):
                ret['target_location'] = self.world_coord_to_minimap([t.target_world_space_pos.x,
                                                                      t.target_world_space_pos.y])
                ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd_pt)]
            if t.HasField('target_unit_tag'):
                ret['target_units'] = [t.target_unit_tag]
                ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd_unit)]
            else:
                ret['action_type'] = [self.ability_to_raw_func(t.ability_id, actions.raw_cmd)]
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
            return ret
        else:
            return None

    def dict2tensor(self, data):
        new_data = {}
        for k, v in data.items():
            if v is None:
                v = 'none'  # for convenience in dataloader
            else:
                v = torch.LongTensor(v)
            new_data[k] = v
        return new_data

    def ability_to_raw_func(self, ability_id, cmd_type):
        if ability_id not in actions.RAW_ABILITY_IDS:
            print("unknown ability id: {}".format(ability_id))
            return 0
        for func in actions.RAW_ABILITY_IDS[ability_id]:
            if func.function_type is cmd_type:
                return func.id
        return 0  # error case, regard as no op
