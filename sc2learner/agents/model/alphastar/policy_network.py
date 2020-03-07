import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitsHead, LocationHead, ActionTypeHead,\
    TargetUnitHead
from .core import CoreLstm
from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder
from sc2learner.nn_utils import fc_block
from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from pysc2.lib.static_data import NUM_UNIT_TYPES, UNIT_TYPES_REORDER, ACTIONS_REORDER_INV
from ..actor_critic.actor_critic import ActorCriticBase


def build_obs_encoder(name):
    obs_encoder_dict = {
        'scalar_encoder': ScalarEncoder,
        'spatial_encoder': SpatialEncoder,
        'entity_encoder': EntityEncoder,
    }
    return obs_encoder_dict[name]


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


class Policy(ActorCriticBase):
    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleDict()
        for item in cfg.obs_encoder.encoder_names:
            self.encoder[item] = build_obs_encoder(item)(cfg.obs_encoder[item])
        self.core_lstm = CoreLstm(cfg.core)
        self.head = nn.ModuleDict()
        for item in cfg.head.head_names:
            self.head[item] = build_head(item)(cfg.head[item])

        self.scatter_project = fc_block(cfg.scatter.input_dim, cfg.scatter.output_dim)
        self.scatter_dim = cfg.scatter.output_dim

    def _scatter_connection(self, spatial_info, entity_embeddings, entity_raw):
        if isinstance(entity_embeddings, collections.Sequence):
            x = [t.squeeze(0) for t in entity_embeddings]
            num_list = [t.shape[0] for t in x]
            x = torch.cat(x, dim=0)
            project_embeddings = self.scatter_project(x)
            project_embeddings = torch.split(project_embeddings, num_list, dim=0)
        else:
            project_embeddings = self.scatter_project(entity_embeddings)
        B, _, H, W = spatial_info.shape
        scatter_map = torch.zeros(B, self.scatter_dim, H, W, device=spatial_info.device)
        for b in range(B):
            N = entity_embeddings[b].shape[0]
            for n in range(N):
                h, w = entity_raw[b]['location'][n]
                h = min(max(0, h), H-1)
                w = min(max(0, w), W-1)
                scatter_map[b, :, h, w] = project_embeddings[b][n]
        return torch.cat([spatial_info, scatter_map], dim=1)

    def _look_up_action_attr(self, action_type, entity_raw, units_num, location_dims=(256, 256)):
        action_mask = {'select_unit_type_mask': [], 'select_unit_mask': [], 'target_unit_type_mask': [],
                       'target_unit_mask': [], 'location_mask': []}
        device = action_type[0].device
        '''
        for idx, action in enumerate(action_type):
            action_mask['select_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
            action_mask['select_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            action_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
            action_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            action_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
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
                action_mask['select_unit_type_mask'].append(select_unit_type_mask.to(device))
                select_unit_mask = torch.zeros(1, units_num[idx])
                for i, t in enumerate(entity_raw[idx]['type']):
                    if t in type_list:
                        select_unit_mask[0, i] = 1
                action_mask['select_unit_mask'].append(select_unit_mask.to(device))
            else:
                action_mask['select_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_mask['select_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            if value['target_units']:
                action_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            else:
                action_mask['target_unit_mask'].append(torch.ones(1, units_num[idx], device=device))
                action_mask['target_unit_type_mask'].append(torch.ones(1, NUM_UNIT_TYPES, device=device))
            if value['target_location']:
                action_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
            else:
                action_mask['location_mask'].append(torch.ones(1, *location_dims, device=device))
            for k in action_attr.keys():
                action_attr[k].append(value[k])
        return action_attr, action_mask

    def _obs_encode(self, inputs):
        embedded_scalar, scalar_context = self.encoder['scalar_encoder'](inputs['scalar_info'])
        entity_embeddings, embedded_entity = self.encoder['entity_encoder'](inputs['entity_info'])
        spatial_input = self._scatter_connection(inputs['spatial_info'], entity_embeddings, inputs['entity_raw'])
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])

        embedded_entity, embedded_spatial, embedded_scalar = (embedded_entity.unsqueeze(0),
                                                              embedded_spatial.unsqueeze(0),
                                                              embedded_scalar.unsqueeze(0))
        lstm_output, next_state = self.core_lstm(
            embedded_entity, embedded_spatial, embedded_scalar, inputs['prev_state'])
        lstm_output = lstm_output.squeeze(0)
        return lstm_output, next_state, entity_embeddings, scalar_context, map_skip

    # overwrite
    def mimic(self, inputs, temperature=1.0):
        '''
            input(keys): scalar_info, entity_info, spatial_info, prev_state, entity_raw, actions
        '''
        lstm_output, next_state, entity_embeddings, scalar_context, map_skip = self._obs_encode(inputs)

        actions = inputs['actions']
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        action_type = torch.LongTensor(actions['action_type']).to(lstm_output.device)
        units_num = [t.shape[0] for t in inputs['entity_info']]

        logits['action_type'], action_type, embeddings = self.head['action_type_head'](
            lstm_output, scalar_context, temperature, action_type)
        action_attr, mask = self._look_up_action_attr(action_type, inputs['entity_raw'], units_num)

        logits['delay'], delay, embeddings = self.head['delay_head'](embeddings)
        for idx in range(action_type.shape[0]):
            embedding = embeddings[idx:idx+1]
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
                    entity_embeddings[idx], temperature, selected_units_num)
                logits['selected_units'].append(logits_selected_units[0])
            if isinstance(actions['target_units'][idx], torch.Tensor):
                if not action_attr['target_units'][idx]:
                    print('target_units', actions['action_type'][idx], actions['target_units'][idx], idx)
                logits_target_units, target_units = self.head['target_unit_head'](
                    embedding, mask['target_unit_type_mask'][idx], mask['target_unit_mask'][idx],
                    entity_embeddings[idx], temperature)
                logits['target_units'].append(logits_target_units)
            if isinstance(actions['target_location'][idx], torch.Tensor):
                if not action_attr['target_location'][idx]:
                    print('target_location', actions['action_type'][idx], actions['target_location'][idx], idx)
                if isinstance(map_skip[0], torch.Tensor):
                    map_skip_single = [t[idx:idx+1] for t in map_skip]
                elif isinstance(map_skip[0], list):
                    map_skip_single = [t[idx] for t in map_skip]
                else:
                    raise TypeError("invalid map_skip element type: {}".format(type(map_skip[0])))
                logits_location, location = self.head['location_head'](
                    embedding, map_skip_single, mask['location_mask'][idx], temperature)
                logits['target_location'].append(logits_location)

        return logits, next_state

    # overwrite
    def _actor_forward(self, inputs, temperature=1.0):
        lstm_output, next_state, entity_embeddings, scalar_context, map_skip = self._obs_encode(inputs)

        B = inputs['spatial_info'].shape[0]
        actions = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        logits = {'queued': [], 'selected_units': [], 'target_units': [], 'target_location': []}
        units_num = [t.shape[0] for t in inputs['entity_info']]

        # action type
        logits['action_type'], action_type, embeddings = self.head['action_type_head'](
            lstm_output, scalar_context, temperature)
        actions['action_type'] = torch.chunk(action_type, B, dim=0)
        action_attr, mask = self._look_up_action_attr(actions['action_type'], inputs['entity_raw'], units_num)

        # action arg delay
        logits['delay'], delay, embeddings = self.head['delay_head'](embeddings)
        actions['delay'] = torch.chunk(delay, B, dim=0)

        for idx in range(B):
            embedding = embeddings[idx:idx+1]
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
                    entity_embeddings[idx], temperature)
                selected_units = selected_units[0]
            else:
                logits_selected_units, selected_units = None, None
            logits['selected_units'].append(logits_selected_units)
            actions['selected_units'].append(selected_units)
            # action arg target_units
            if action_attr['target_units'][idx]:
                logits_target_units, target_units = self.head['target_unit_head'](
                    embedding, mask['target_unit_type_mask'][idx], mask['target_unit_mask'][idx],
                    entity_embeddings[idx], temperature)
                target_units = target_units[0]
            else:
                logits_target_units, target_units = None, None
            logits['target_units'].append(logits_target_units)
            actions['target_units'].append(target_units)
            # action arg target_location
            if action_attr['target_location'][idx]:
                map_skip_single = [t[idx:idx+1] for t in map_skip]
                logits_location, location = self.head['location_head'](
                    embedding, map_skip_single, mask['location_mask'][idx], temperature)
            else:
                logits_location, location = None, None
            logits['target_location'].append(logits_location)
            actions['target_location'].append(location)

        return actions, logits, next_state

    # overwrite
    def evaluate(self, inputs, **kwargs):
        '''
            batch size = 1
        '''
        ratio = self.cfg.location_expand_ratio
        Y, X = inputs['map_size'][0]

        actions, _, next_state = self._actor_forward(inputs, **kwargs)

        if isinstance(actions['target_location'][0], torch.Tensor):
            location = actions['target_location'][0]
            transformed_location = torch.cat([location // (ratio*X), location % (ratio*X)], 0)
            transformed_location = transformed_location.float().div(ratio)
            actions['target_location'] = [transformed_location]

        # error action(no necessary selected units)
        if isinstance(actions['selected_units'][0], torch.Tensor) and actions['selected_units'][0].shape[0] == 0:
            device = actions['action_type'][0].device
            actions = {'action_type': [torch.LongTensor([0]).to(device)], 'delay': [torch.LongTensor([0]).to(device)],
                       'queued': [None], 'selected_units': [None], 'target_units': [None], 'target_location': [None]}
        return {
            'actions': actions,
            'next_state': next_state
        }
