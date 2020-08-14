import collections

import torch
import torch.nn as nn

from nervex.torch_utils import fc_block, build_activation
from .core import CoreLstm
from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder


def build_obs_encoder(name):
    obs_encoder_dict = {
        'scalar_encoder': ScalarEncoder,
        'spatial_encoder': SpatialEncoder,
        'entity_encoder': EntityEncoder,
    }
    return obs_encoder_dict[name]


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleDict()
        for item in cfg.obs_encoder.encoder_names:
            self.encoder[item] = build_obs_encoder(item)(cfg.obs_encoder[item])
        self.core_lstm = CoreLstm(cfg.core_lstm)

        self.scatter_project = fc_block(cfg.scatter.input_dim, cfg.scatter.output_dim)
        self.scatter_dim = cfg.scatter.output_dim
        self.use_score_cumulative = cfg.obs_encoder.use_score_cumulative
        if self.use_score_cumulative:
            self.score_cumulative_encoder = fc_block(
                self.cfg.score_cumulative.input_dim,
                self.cfg.score_cumulative.output_dim,
                activation=build_activation(self.cfg.score_cumulative.activation)
            )

    def _scatter_connection(self, spatial_info, entity_embeddings, entity_raw):
        if isinstance(entity_embeddings, collections.abc.Sequence):
            x = [t.squeeze(0) for t in entity_embeddings]
            num_list = [t.shape[0] for t in x]
            x = torch.cat(x, dim=0)
            project_embeddings = self.scatter_project(x)
            project_embeddings = torch.split(project_embeddings, num_list, dim=0)
        else:
            project_embeddings = self.scatter_project(entity_embeddings)
        B, _, H, W = spatial_info.shape
        device = spatial_info.device
        scatter_map = torch.zeros(B, self.scatter_dim, H * W, device=device)
        for b in range(B):
            N = entity_embeddings[b].shape[0]
            index = torch.LongTensor(entity_raw[b]['location']).to(device)
            index[:, 0].clamp_(0, H - 1)
            index[:, 1].clamp_(0, W - 1)
            index = index[:, 0] * W + index[:, 1]
            index = index.unsqueeze(0).repeat(self.scatter_dim, 1)
            src = project_embeddings[b].permute(1, 0)
            scatter_map[b].scatter_(dim=1, index=index, src=src)
        scatter_map = scatter_map.reshape(B, self.scatter_dim, H, W)
        return torch.cat([spatial_info, scatter_map], dim=1)

    def forward(self, inputs):
        '''
        Arguments:
            - inputs:
            dict with field:
                - scalar_info
                - spatial_info
                - entity_raw
                - entity_info
                - map_size
                - prev_state
                - score_cumulative
        Outputs:
            - lstm_output: The LSTM state for the next step. Tensor of size [seq_len, batch_size, hidden_size]
            - next_state: The LSTM state for the next step.
              As list [H,C], H and C are of size [num_layers, batch_size, hidden_size]
            - entity_embeddings: The embedding of each entity. Tensor of size [batch_size, entity_num, output_dim]
            - map_skip
            - scalar_context
            - baseline_feature
            - cum_stat: OrderedDict of various cumulative_statistics
            - socre_embedding: score cumulative embedding for baseline
        '''
        embedded_scalar, scalar_context, baseline_feature, cum_stat = self.encoder['scalar_encoder'](
            inputs['scalar_info']
        )
        entity_embeddings, embedded_entity = self.encoder['entity_encoder'](inputs['entity_info'])
        spatial_input = self._scatter_connection(inputs['spatial_info'], entity_embeddings, inputs['entity_raw'])
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])

        embedded_entity, embedded_spatial, embedded_scalar = (
            embedded_entity.unsqueeze(0), embedded_spatial.unsqueeze(0), embedded_scalar.unsqueeze(0)
        )
        lstm_output, next_state = self.core_lstm(
            embedded_entity, embedded_spatial, embedded_scalar, inputs['prev_state']
        )
        lstm_output = lstm_output.squeeze(0)
        if self.use_score_cumulative:
            score_embedding = self.score_cumulative_encoder(inputs['scalar_info']['score_cumulative'])
        else:
            score_embedding = None  # placeholder
        return lstm_output, next_state, entity_embeddings, map_skip, scalar_context, inputs[
            'spatial_info'], baseline_feature, cum_stat, score_embedding

    def encode_parallel_forward(self, inputs):
        embedded_scalar, scalar_context, baseline_feature, cum_stat = self.encoder['scalar_encoder'](
            inputs['scalar_info']
        )
        entity_embeddings, embedded_entity = self.encoder['entity_encoder'](inputs['entity_info'])
        spatial_input = self._scatter_connection(inputs['spatial_info'], entity_embeddings, inputs['entity_raw'])
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])
        return [
            embedded_entity, embedded_spatial, embedded_scalar, scalar_context, baseline_feature, cum_stat,
            entity_embeddings, map_skip
        ]
