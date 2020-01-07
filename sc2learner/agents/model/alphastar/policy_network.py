import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import DelayHead, QueuedHead, SelectedUnitsHead, TargetUnitsHead, LocationHead, ActionTypeHead
from .core import CoreLstm
from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder
from sc2learner.utils import fc_block


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
        'location_head': LocationHead,
    }
    return head_dict[name]


class Policy(nn.Module):
    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.encoder = nn.ModuleDict()
        for item in cfg.obs_encoder.encoder_names:
            self.encoder[item] = build_obs_encoder(item)(cfg.obs_encoder[item])
        self.core_lstm = CoreLstm(cfg.core)
        self.head = nn.ModuleDict()
        for item in cfg.obs_encoder.head_names:
            self.head[item] = build_head(item)(cfg.head[item])

        self.scatter_project = fc_block(cfg.scatter.input_dim, cfg.scatter.output_dim)

    def _scatter_connection(self, spatial_info, entity_embeddings, entity_location):
        project_embeddings = self.scatter_project(entity_embeddings)
        _, N, C = project_embeddings.shape
        B, _, H, W = spatial_info.shape
        scatter_map = torch.zeros(B, C, H, W, device=spatial_info.device)
        for b in range(B):
            for n in range(N):
                h, w = entity_location[b, n]
                scatter_map[b, :, h, w] = project_embeddings[b, n]
        return torch.cat([spatial_info, scatter_map], dim=1)

    def _look_up_action_attr(self, action_type, inputs):
        raise NotImplementedError

    def forward(self, inputs, temperature):
        '''
            input(keys): scalar_info, entity_info, spatial_info, prev_state, entity_location
        '''
        embedded_scalar, scalar_context = self.encoder['scalar_encoder'](inputs['scalar_info'])
        entity_embeddings, embedded_entity = self.encoder['entity_encoder'](inputs['entity_info'])
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](
            self._scatter_connection(inputs['spatial_info'], entity_embeddings, inputs['entity_location']))

        lstm_output, next_state = self.core_lstm(
            embedded_entity, embedded_spatial, embedded_scalar, inputs['prev_state'])

        action_type_logits, action_type, embedding = self.head['action_type_head'](
            lstm_output, scalar_context, temperature)
        actions = [{'type': a, 'type_logits': logits} for a, logits in zip(action_type, action_type_logits)]
        for item in actions:
            action_attr = self._look_up_action_attr(item['type'], inputs)
            item['delay_logits'], item['delay'], embedding = self.head['delay_head'](embedding)
            if action_attr['enable_queued']:
                item['queued_logits'], item['queued'], embedding = self.head['queued_head'](embedding, temperature)
            if action_attr['enable_select_units']:
                item['units_logits'], item['units'], embedding = self.head['selected_units_head'](
                    embedding, action_attr['unit_type_mask'], action_attr['units_mask'], entity_embeddings, temperature)
            if action_attr['enable_target_unit']:
                item['target_unit_logits'], item['target_unit'] = self.head['target_units_head'](
                    embedding, action_attr['unit_type_mask'], action_attr['units_mask'], entity_embeddings, temperature)
            if action_attr['entity_location']:
                item['location_logits'], item['location'] = self.head['location_head'](
                    embedding, map_skip, action_attr['location_mask'], temperature)

        return actions, next_state
