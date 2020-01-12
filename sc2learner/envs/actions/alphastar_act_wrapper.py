import torch


class AlphastarActParser(object):
    def __init__(self):
        self.input_template = {'camera_move': self._parse_raw_camera_move,
                               'unit_command': self._parse_raw_unit_command}
        self.output_template = ['action_type', 'delay', 'queued', 'selected_units', 'target_units', 'target_location']

    def parse(self, action):
        ret = {k: None for k in self.output_template}
        update_count = {}
        for k, f in self.input_template.items():
            act_val = getattr(action, k)
            v = f(act_val)
            if v is not None:
                ret.update(v)
                update_count[k] = 1
        assert(sum(update_count.values()) == 1)  # only one item has value
        return ret

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_camera_move(self, t):
        if t.HasField('center_world_space'):
            location = [t.center_world_space.x, t.center_world_space.y]
            return {'action_type': [168], 'target_location': location}  # raw_camera_move 168
        else:
            return None

    # refer to https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto
    def _parse_raw_unit_command(self, t):
        if t.HasField('ability_id'):
            ret = {'action_type': [t.ability_id], 'selected_units': [t.unit_tags]}
            if t.HasField('queue_command'):
                ret['queued'] = [t.queue_command]
            assert((t.HasField('target_world_space_pos')) + (t.HasField('target_unit_tag')) == 1)
            if t.HasField('target_world_space_pos'):
                ret['target_location'] = [t.target_world_space_pos.x, t.target_world_space_pos.y]
            if t.HasField('target_unit_tag'):
                ret['target_unit_tag'] = [t.target_unit_tag]
            return ret
        else:
            return None

    def dict2tensor(self, data, dtype=torch.long):
        new_data = {}
        for k, v in data.items():
            if v is None:
                v = 'none'  # for convenience in dataloader
            else:
                v = torch.FloatTensor(v)
            new_data[k] = v
        return new_data
