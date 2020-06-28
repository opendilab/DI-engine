from collections import namedtuple
import copy

import torch
import torch.nn as nn

from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK, ACTIONS_STAT
from pysc2.lib.static_data import NUM_UNIT_TYPES, UNIT_TYPES_REORDER, ACTIONS_REORDER_INV, PART_ACTIONS_MAP,\
    PART_ACTIONS_MAP_INV
from sc2learner.envs import get_location_mask
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitHead, LocationHead, ActionTypeHead


def build_head(name):
    head_dict = {
        'action_type_head': ActionTypeHead,
        'base_action_type_head': ActionTypeHead,
        'spec_action_type_head': ActionTypeHead,
        'delay_head': DelayHead,
        'queued_head': QueuedHead,
        'selected_units_head': SelectedUnitsHead,
        'target_unit_head': TargetUnitHead,
        'location_head': LocationHead,
    }
    return head_dict[name]


class Policy(nn.Module):
    MimicInput = namedtuple(
        'MimicInput', [
            'actions', 'entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip',
            'scalar_context', 'spatial_info'
        ]
    )

    EvaluateInput = namedtuple(
        'EvaluateInput', [
            'entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip', 'scalar_context',
            'spatial_info'
        ]
    )

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.cfg = cfg
        self.head = nn.ModuleDict()
        for item in cfg.head.head_names:
            self.head[item] = build_head(item)(cfg.head[item])

    def _look_up_action_attr(self, action_type, entity_raw, units_num, spatial_info):
        action_arg_mask = {
            'select_unit_type_mask': [],
            'select_unit_mask': [],
            'target_unit_type_mask': [],
            'target_unit_mask': [],
            'location_mask': []
        }
        device = action_type[0].device
        action_attr = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        for idx, action in enumerate(action_type):
            action_type_val = ACTIONS_REORDER_INV[action.item()]
            action_info_hard_craft = GENERAL_ACTION_INFO_MASK[action_type_val]
            try:
                action_info_stat = ACTIONS_STAT[action_type_val]
            except KeyError as e:
                print('We are issuing a command (reordered:{}), never seen in replays'.format(action_type_val))
                action_info_stat = {'selected_type': [], 'target_type': []}
            # else case is the placeholder
            if action_info_hard_craft['selected_units']:
                type_hard_craft = set(action_info_hard_craft['avail_unit_type_id'])
                type_stat = set(action_info_stat['selected_type'])
                type_set = type_hard_craft.union(type_stat)
                reorder_type_list = [UNIT_TYPES_REORDER[t] for t in type_set]
                select_unit_type_mask = torch.zeros(NUM_UNIT_TYPES)
                select_unit_type_mask[reorder_type_list] = 1
                action_arg_mask['select_unit_type_mask'].append(select_unit_type_mask.to(device))
                select_unit_mask = torch.zeros(units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_set:
                        select_unit_mask[i] = 1
                action_arg_mask['select_unit_mask'].append(select_unit_mask.to(device))
            else:
                action_arg_mask['select_unit_mask'].append(torch.zeros(units_num[idx]).to(device))
                action_arg_mask['select_unit_type_mask'].append(torch.zeros(NUM_UNIT_TYPES).to(device))
            if action_info_hard_craft['target_units']:
                type_set = set(action_info_stat['target_type'])
                reorder_type_list = [UNIT_TYPES_REORDER[t] for t in type_set]
                target_unit_type_mask = torch.zeros(NUM_UNIT_TYPES)
                target_unit_type_mask[reorder_type_list] = 1
                action_arg_mask['target_unit_type_mask'].append(target_unit_type_mask.to(device))
                target_unit_mask = torch.zeros(units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_set:
                        target_unit_mask[i] = 1
                action_arg_mask['target_unit_mask'].append(target_unit_mask.to(device))
            else:
                action_arg_mask['target_unit_mask'].append(torch.zeros(units_num[idx]).to(device))
                action_arg_mask['target_unit_type_mask'].append(torch.zeros(NUM_UNIT_TYPES).to(device))
            # TODO(nyz) location mask for different map size
            if action_info_hard_craft['target_location']:
                location_mask = get_location_mask(action_type_val, spatial_info[idx])
                action_arg_mask['location_mask'].append(location_mask)
            else:
                shapes = spatial_info[idx].shape[-2:]
                action_arg_mask['location_mask'].append(torch.zeros(1, *shapes).to(device))
            # get action attribute(which args the action type owns)
            for k in action_attr.keys():
                action_attr[k].append(action_info_hard_craft[k])
        # stack mask
        for k in ['select_unit_type_mask', 'target_unit_type_mask', 'location_mask']:
            action_arg_mask[k] = torch.stack(action_arg_mask[k], dim=0)
        return action_attr, action_arg_mask

    def _action_type_forward(self, lstm_output, scalar_context, action_type_mask, temperature, action_type=None):
        kwargs = {
            'lstm_output': lstm_output,
            'scalar_context': scalar_context,
            'action_type_mask': action_type_mask,
            'temperature': temperature,
            'action_type': action_type
        }
        if 'action_type_head' in self.head.keys():
            return self.head['action_type_head'](**kwargs)
        elif 'base_action_type_head' in self.head.keys() and 'spec_action_type_head' in self.head.keys():
            # get part action mask
            base_action_type_mask = action_type_mask[:, list(PART_ACTIONS_MAP['base'].keys())]
            spec_action_type_mask = action_type_mask[:, list(PART_ACTIONS_MAP['spec'].keys())]
            if action_type is not None:
                base_action_type = action_type.clone()
                spec_action_type = action_type.clone()
                # to part action type id
                for idx, val in enumerate(action_type):
                    val = val.item()
                    if val == 0:
                        continue
                    elif val in PART_ACTIONS_MAP['base'].keys():
                        spec_action_type[idx] = 0
                        base_action_type[idx] = PART_ACTIONS_MAP['base'][val]
                    else:
                        spec_action_type[idx] = PART_ACTIONS_MAP['spec'][val]
                        base_action_type[idx] = 0
                # double head forward
                kwargs['action_type'] = base_action_type
                kwargs['action_type_mask'] = base_action_type_mask
                base_logits, base_action_type, base_embeddings = self.head['base_action_type_head'](**kwargs)
                kwargs['action_type'] = spec_action_type
                kwargs['action_type_mask'] = spec_action_type_mask
                spec_logits, spec_action_type, spec_embeddings = self.head['spec_action_type_head'](**kwargs)
            else:
                kwargs['action_type_mask'] = base_action_type_mask
                base_logits, base_action_type, base_embeddings = self.head['base_action_type_head'](**kwargs)
                kwargs['action_type_mask'] = spec_action_type_mask
                spec_logits, spec_action_type, spec_embeddings = self.head['spec_action_type_head'](**kwargs)
            # to total action type id
            for idx, val in enumerate(base_action_type):
                base_action_type[idx] = PART_ACTIONS_MAP_INV['base'][val.item()]
            for idx, val in enumerate(spec_action_type):
                spec_action_type[idx] = PART_ACTIONS_MAP_INV['spec'][val.item()]
            mask = torch.where(
                spec_action_type == 0, torch.ones_like(spec_action_type), torch.zeros_like(spec_action_type)
            )  # noqa
            action_type = mask * base_action_type + spec_action_type
            mask = mask.view(-1, *[1 for _ in range(len(base_embeddings.shape) - 1)]).to(
                base_embeddings.dtype
            )  # batch is the first dim  # noqa
            embeddings = mask * base_embeddings + (1 - mask) * spec_embeddings
            return [base_logits, spec_logits], action_type, embeddings
        else:
            raise KeyError("no necessary action type head in heads{}".format(self.head.keys()))

    def mimic(self, inputs, temperature=1.0):
        '''
            Overview: supervised learning policy forward graph
            Arguments:
                - inputs (:obj:`Policy.Input`) namedtuple
                - temperature (:obj:`float`) logits sample temperature
            Returns:
                - logits (:obj:`dict`) logits(or other format) for calculating supervised learning loss
        '''
        def to_tensor_action(actions):
            actions = copy.deepcopy(actions)
            device = actions['action_type'][0].device
            for k, v in actions.items():
                if k in ['action_type', 'delay']:
                    actions[k] = torch.cat(v, dim=0)
                elif k in ['queued']:
                    new_v = []
                    for t in v:
                        if isinstance(t, torch.Tensor):
                            new_v.append(t)
                        else:
                            new_v.append(torch.zeros(1, dtype=torch.long, device=device))  # set no_queued to not_queued
                    actions[k] = torch.cat(new_v, dim=0)
                elif k in ['selected_units']:
                    new_v = []
                    for t in v:
                        if isinstance(t, torch.Tensor):
                            new_v.append(t)
                        else:
                            new_v.append(torch.LongTensor([]).to(device))
                    actions[k] = new_v
                elif k in ['target_units', 'target_location']:
                    pass
                else:
                    raise KeyError(k)
            return actions

        actions, entity_raw, action_type_mask, lstm_output,\
            entity_embeddings, map_skip, scalar_context, spatial_info = inputs
        actions = to_tensor_action(actions)
        B = len(entity_raw)
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        units_num = [len(t['id']) for t in entity_raw]

        # action type
        logits['action_type'], _, embeddings = self._action_type_forward(
            lstm_output, scalar_context, action_type_mask, temperature, actions['action_type']
        )
        action_attr, mask = self._look_up_action_attr(actions['action_type'], entity_raw, units_num, spatial_info)

        # action arg delay
        logits['delay'], _, embeddings = self.head['delay_head'](embeddings, actions['delay'])

        # action arg queued
        logits_queued, _, embeddings = self.head['queued_head'](embeddings, temperature, actions['queued'])
        logits['queued'] = self._mask_select([logits_queued], action_attr['queued'])

        logits_selected_units, _, embeddings = self.head['selected_units_head'](
            embeddings, mask['select_unit_type_mask'], mask['select_unit_mask'], entity_embeddings, temperature,
            actions['selected_units']
        )
        logits['selected_units'] = self._mask_select([logits_selected_units], action_attr['selected_units'])

        logits_target_units, _ = self.head['target_unit_head'](
            embeddings,
            mask['target_unit_type_mask'],
            mask['target_unit_mask'],
            entity_embeddings,
            temperature,
        )
        logits['target_units'] = self._mask_select([logits_target_units], action_attr['target_units'])

        logits_location, _ = self.head['location_head'](embeddings, map_skip, mask['location_mask'], temperature)
        logits['target_location'] = self._mask_select([logits_location], action_attr['target_location'])

        return logits

    def evaluate(self, inputs, temperature=1.0):
        '''
            Overview: agent(policy) evaluate forward graph, or in reinforcement learning
            Arguments:
                - inputs (:obj:`Policy.Input`) namedtuple
                - temperature (:obj:`float`) logits sample temperature
            Returns:
                - logits (:obj:`dict`) logits
                - actions (:obj:`dict`) actions predicted by agent(policy)
        '''
        entity_raw, action_type_mask, lstm_output, entity_embeddings, map_skip, scalar_context, spatial_info = inputs

        B = len(entity_raw)
        actions = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        units_num = [len(t['id']) for t in entity_raw]

        # action type
        logits['action_type'], action_type, embeddings = self._action_type_forward(
            lstm_output, scalar_context, action_type_mask, temperature
        )
        actions['action_type'] = torch.chunk(action_type, B, dim=0)
        action_attr, mask = self._look_up_action_attr(actions['action_type'], entity_raw, units_num, spatial_info)

        # action arg delay
        logits['delay'], delay, embeddings = self.head['delay_head'](embeddings)
        actions['delay'] = torch.chunk(delay, B, dim=0)

        logits_queued, queued, embeddings = self.head['queued_head'](embeddings, temperature)
        logits['queued'], actions['queued'] = self._mask_select([logits_queued, queued], action_attr['queued'])

        logits_selected_units, selected_units, embeddings = self.head['selected_units_head'](
            embeddings, mask['select_unit_type_mask'], mask['select_unit_mask'], entity_embeddings, temperature
        )
        logits['selected_units'], actions['selected_units'] = self._mask_select(
            [logits_selected_units, selected_units], action_attr['selected_units']
        )

        logits_target_units, target_units = self.head['target_unit_head'](
            embeddings, mask['target_unit_type_mask'], mask['target_unit_mask'], entity_embeddings, temperature
        )
        logits['target_units'], actions['target_units'] = self._mask_select(
            [logits_target_units, target_units], action_attr['target_units']
        )

        logits_location, location = self.head['location_head'](embeddings, map_skip, mask['location_mask'], temperature)
        logits['target_location'], actions['target_location'] = self._mask_select(
            [logits_location, location], action_attr['target_location']
        )

        actions = self._get_action_entity_raw(actions, entity_raw)

        return actions, logits

    def _mask_select(self, data, mask):
        def chunk(item):
            return torch.chunk(item, item.shape[0], 0) if isinstance(item, torch.Tensor) else item

        assert isinstance(data, list)
        data = [list(chunk(d)) for d in data]
        for idx, m in enumerate(mask):
            if not m:
                for i in range(len(data)):
                    data[i][idx] = None
        if len(data) == 1:
            return data[0]
        else:
            return data

    def _get_action_entity_raw(self, actions, entity_raw):
        B = len(entity_raw)
        action_entity_raw = []
        for b in range(B):
            selected_units = actions['selected_units'][b]
            target_units = actions['target_units'][b]
            selected_units = [] if selected_units is None else selected_units.tolist()
            target_units = [] if target_units is None else target_units.tolist()
            units = list(set(selected_units).union(set(target_units)))

            entity_raw_per_frame = entity_raw[b]
            keys = entity_raw_per_frame.keys()
            action_entity_raw_per_frame = {}
            for u in units:
                try:
                    entity_raw_u = {k: entity_raw_per_frame[k][u] for k in keys}
                except IndexError:
                    entity_raw_u = {k: entity_raw_per_frame[k][-1] for k in keys}
                    print('[ERROR]: invalid unit idx: {} {} {}'.format(u, selected_units, target_units))
                    print('[ERROR]: len entity_raw_per_frame: {}'.format(len(entity_raw_per_frame['id'])))
                action_entity_raw_per_frame[u] = entity_raw_u
            action_entity_raw.append(action_entity_raw_per_frame)

        actions['action_entity_raw'] = action_entity_raw
        return actions

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['mimic', 'evaluate'])
        return getattr(self, mode)(inputs, **kwargs)
