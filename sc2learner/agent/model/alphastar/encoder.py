import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import CoreLstm
from .obs_encoder import ScalarEncoder, SpatialEncoder, EntityEncoder
from sc2learner.torch_utils import fc_block


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
        Outputs:
            - lstm_output: The LSTM state for the next step. Tensor of size [seq_len, batch_size, hidden_size]
            - next_state: The LSTM state for the next step.
              As list [H,C], H and C are of size [num_layers, batch_size, hidden_size]
            - entity_embeddings: The embedding of each entity. Tensor of size [batch_size, entity_num, output_dim]
            - map_skip
            - scalar_context
            - baseline_feature
            - cum_stat: OrderedDict of various cumulative_statistics
        '''
        embedded_scalar, scalar_context, baseline_feature, cum_stat = self.encoder['scalar_encoder'](inputs['scalar_info'])  # noqa
        entity_embeddings, embedded_entity = self.encoder['entity_encoder'](inputs['entity_info'])
        spatial_input = self._scatter_connection(inputs['spatial_info'], entity_embeddings, inputs['entity_raw'])
        embedded_spatial, map_skip = self.encoder['spatial_encoder'](spatial_input, inputs['map_size'])

        embedded_entity, embedded_spatial, embedded_scalar = (embedded_entity.unsqueeze(0),
                                                              embedded_spatial.unsqueeze(0),
                                                              embedded_scalar.unsqueeze(0))
        lstm_output, next_state = self.core_lstm(
            embedded_entity, embedded_spatial, embedded_scalar, inputs['prev_state'])
        lstm_output = lstm_output.squeeze(0)
        return lstm_output, next_state, entity_embeddings, map_skip, scalar_context, baseline_feature, cum_stat
