from collections import namedtuple

import torch
import torch.nn as nn

from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from pysc2.lib.static_data import NUM_UNIT_TYPES, UNIT_TYPES_REORDER, ACTIONS_REORDER_INV
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitsHead, LocationHead, ActionTypeHead, \
    TargetUnitHead


def build_head(name):
    head_dict = {
        'action_type_head': ActionTypeHead,
        'delay_head': DelayHead,
        'queued_head': QueuedHead,
        'selected_units_head': SelectedUnitsHead,
        'target_units_head': TargetUnitsHead,
        'target_unit_head': TargetUnitHead,
        'location_head': LocationHead,
    }
    return head_dict[name]


class Policy(nn.Module):
    MimicInput = namedtuple(
        'MimicInput',
        ['actions', 'entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip', 'scalar_context']
    )

    EvaluateInput = namedtuple(
        'EvaluateInput',
        ['entity_raw', 'action_type_mask', 'lstm_output', 'entity_embeddings', 'map_skip', 'scalar_context']
    )

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.cfg = cfg
        self.head = nn.ModuleDict()
        for item in cfg.head.head_names:
            self.head[item] = build_head(item)(cfg.head[item])

    def _look_up_action_attr(self, action_type, entity_raw, units_num, location_dims=(256, 256)):
        action_arg_mask = {
            'select_unit_type_mask': [],
            'select_unit_mask': [],
            'target_unit_type_mask': [],
            'target_unit_mask': [],
            'location_mask': []
        }
        device = action_type[0].device
        '''
        for idx, action in enumerate(action_type):
            action_arg_mask['select_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
            action_arg_mask['select_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            action_arg_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
            action_arg_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            action_arg_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
        action_attr = {'queued': 'none', 'selected_units': 'none', 'target_units': 'none', 'target_location': 'none'}
        '''
        action_attr = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        for idx, action in enumerate(action_type):
            value = GENERAL_ACTION_INFO_MASK[ACTIONS_REORDER_INV[action.item()]]
            if value['selected_units']:
                type_list = value['avail_unit_type_id']
                reorder_type_list = [UNIT_TYPES_REORDER[t] for t in type_list]
                select_unit_type_mask = torch.zeros(1, NUM_UNIT_TYPES)
                select_unit_type_mask[:, reorder_type_list] = 1
                action_arg_mask['select_unit_type_mask'].append(select_unit_type_mask.to(device))
                select_unit_mask = torch.zeros(1, units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_list:
                        select_unit_mask[0, i] = 1
                action_arg_mask['select_unit_mask'].append(select_unit_mask.to(device))
            else:
                action_arg_mask['select_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_arg_mask['select_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            if value['target_units']:
                action_arg_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_arg_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            else:
                action_arg_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_arg_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            if value['target_location']:
                action_arg_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
            else:
                action_arg_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
            for k in action_attr.keys():
                action_attr[k].append(value[k])
        return action_attr, action_arg_mask

    def mimic(self, inputs, temperature=1.0):
        '''
            Overview: supervised learning policy forward graph
            Arguments:
                - inputs (:obj:`Policy.Input`) namedtuple
                - temperature (:obj:`float`) logits sample temperature
            Returns:
                - logits (:obj:`dict`) logits(or other format) for calculating supervised learning loss
        '''
        actions, entity_raw, action_type_mask, lstm_output, entity_embeddings, map_skip, scalar_context = inputs

        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        action_type = torch.LongTensor(actions['action_type']).to(lstm_output.device)
        units_num = [len(t['id']) for t in entity_raw]

        logits['action_type'], action_type, embeddings = self.head['action_type_head'](
            lstm_output, scalar_context, action_type_mask, temperature, action_type
        )
        action_attr, mask = self._look_up_action_attr(action_type, entity_raw, units_num)

        logits['delay'], delay, embeddings = self.head['delay_head'](embeddings)
        for idx in range(action_type.shape[0]):
            embedding = embeddings[idx:idx + 1]
            if isinstance(actions['queued'][idx], torch.Tensor):
                if not action_attr['queued'][idx]:
                    print('queued', actions['action_type'][idx], actions['queued'][idx], idx)
                logits_queued, queued, embedding = self.head['queued_head'](embedding, temperature)
                logits['queued'].append(logits_queued)
            if isinstance(actions['selected_units'][idx], torch.Tensor):
                if not action_attr['selected_units'][idx]:
                    print('selected_units', actions['action_type'][idx], actions['selected_units'][idx], idx)
                selected_units_num = [actions['selected_units'][idx].shape[0]]
                logits_selected_units, selected_units, embedding = self.head['selected_units_head'](
                    embedding, mask['select_unit_type_mask'][idx], mask['select_unit_mask'][idx],
                    entity_embeddings[idx], temperature, selected_units_num
                )
                logits['selected_units'].append(logits_selected_units[0])
            if isinstance(actions['target_units'][idx], torch.Tensor):
                if not action_attr['target_units'][idx]:
                    print('target_units', actions['action_type'][idx], actions['target_units'][idx], idx)
                logits_target_units, target_units = self.head['target_unit_head'](
                    embedding, mask['target_unit_type_mask'][idx], mask['target_unit_mask'][idx],
                    entity_embeddings[idx], temperature
                )
                logits['target_units'].append(logits_target_units)
            if isinstance(actions['target_location'][idx], torch.Tensor):
                if not action_attr['target_location'][idx]:
                    print('target_location', actions['action_type'][idx], actions['target_location'][idx], idx)
                if isinstance(map_skip[0], torch.Tensor):
                    map_skip_single = [t[idx:idx + 1] for t in map_skip]
                elif isinstance(map_skip[0], list):
                    map_skip_single = [t[idx] for t in map_skip]
                else:
                    raise TypeError("invalid map_skip element type: {}".format(type(map_skip[0])))
                logits_location, location = self.head['location_head'](
                    embedding, map_skip_single, mask['location_mask'][idx], temperature
                )
                logits['target_location'].append(logits_location)

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
        entity_raw, lstm_output, action_type_mask, entity_embeddings, map_skip, scalar_context = inputs

        B = len(entity_raw)
        actions = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        units_num = [len(t['id']) for t in entity_raw]

        # action type
        logits['action_type'], action_type, embeddings = self.head['action_type_head'](
            lstm_output, scalar_context, action_type_mask, temperature
        )
        actions['action_type'] = torch.chunk(action_type, B, dim=0)
        action_attr, mask = self._look_up_action_attr(actions['action_type'], entity_raw, units_num)

        # action arg delay
        logits['delay'], delay, embeddings = self.head['delay_head'](embeddings)
        actions['delay'] = torch.chunk(delay, B, dim=0)

        for idx in range(B):
            embedding = embeddings[idx:idx + 1]
            # action arg queued
            if action_attr['queued'][idx]:
                logits_queued, queued, embedding = self.head['queued_head'](embedding, temperature)
            else:
                logits_queued, queued = None, None
            logits['queued'].append(logits_queued)
            actions['queued'].append(queued)
            # action arg selected_units
            if action_attr['selected_units'][idx]:
                logits_selected_units, selected_units, embedding = self.head['selected_units_head'](
                    embedding, mask['select_unit_type_mask'][idx], mask['select_unit_mask'][idx],
                    entity_embeddings[idx], temperature
                )
                selected_units = selected_units[0]
            else:
                logits_selected_units, selected_units = None, None
            logits['selected_units'].append(logits_selected_units)
            actions['selected_units'].append(selected_units)
            # action arg target_units
            if action_attr['target_units'][idx]:
                logits_target_units, target_units = self.head['target_unit_head'](
                    embedding, mask['target_unit_type_mask'][idx], mask['target_unit_mask'][idx],
                    entity_embeddings[idx], temperature
                )
                target_units = target_units[0]
            else:
                logits_target_units, target_units = None, None
            logits['target_units'].append(logits_target_units)
            actions['target_units'].append(target_units)
            # action arg target_location
            if action_attr['target_location'][idx]:
                map_skip_single = [t[idx:idx + 1] for t in map_skip]
                logits_location, location = self.head['location_head'](
                    embedding, map_skip_single, mask['location_mask'][idx], temperature
                )
            else:
                logits_location, location = None, None
            logits['target_location'].append(logits_location)
            actions['target_location'].append(location)

        return actions, logits

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['mimic', 'evaluate'])
        return getattr(self, mode)(inputs, **kwargs)
