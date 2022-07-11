from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import fc_block, build_activation, Transformer
from .entity_encoder import get_binary_embed_mat


def compute_denominator(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x // 2 * 2
    x = torch.div(x, dim)
    x = torch.pow(10000., x)
    x = torch.div(1., x)
    return x


class BeginningBuildOrderEncoder(nn.Module):

    def __init__(self, cfg):
        super(BeginningBuildOrderEncoder, self).__init__()
        self.cfg = cfg
        self.output_dim = self.cfg.output_dim
        self.input_dim = self.cfg.action_one_hot_dim + 20 + self.cfg.binary_dim * 2
        self.act = build_activation(self.cfg.activation, inplace=True)
        self.transformer = Transformer(
            input_dim=self.input_dim,
            head_dim=self.cfg.head_dim,
            hidden_dim=self.cfg.output_dim * 2,
            output_dim=self.cfg.output_dim
        )
        self.embedd_fc = fc_block(self.cfg.output_dim, self.output_dim, activation=self.act)
        self.action_one_hot = nn.Embedding.from_pretrained(
            torch.eye(self.cfg.action_one_hot_dim), freeze=True, padding_idx=None
        )
        self.order_one_hot = nn.Embedding.from_pretrained(torch.eye(20), freeze=True, padding_idx=None)
        self.location_binary = nn.Embedding.from_pretrained(
            get_binary_embed_mat(self.cfg.binary_dim), freeze=True, padding_idx=None
        )

    def _add_seq_info(self, x):
        indices_one_hot = torch.zeros(size=(x.shape[1], x.shape[1]), device=x.device)
        indices = torch.arange(x.shape[1], device=x.device).unsqueeze(dim=1)
        indices_one_hot = indices_one_hot.scatter_(dim=-1, index=indices, value=1.)
        indices_one_hot = indices_one_hot.unsqueeze(0).repeat(x.shape[0], 1, 1)  # expand to batch dim
        return torch.cat([x, indices_one_hot], dim=2)

    def forward(self, x, bo_location):
        x = self.action_one_hot(x.long())
        x = self._add_seq_info(x)
        location_x = bo_location % self.cfg.spatial_x
        location_y = bo_location // self.cfg.spatial_x
        location_x = self.location_binary(location_x.long())
        location_y = self.location_binary(location_y.long())
        x = torch.cat([x, location_x, location_y], dim=2)
        assert len(x.shape) == 3
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.embedd_fc(x)
        return x


class ScalarEncoder(nn.Module):

    def __init__(self, cfg):
        super(ScalarEncoder, self).__init__()
        self.cfg = cfg
        self.act = build_activation(self.cfg.activation, inplace=True)
        self.keys = []
        self.scalar_context_keys = []
        self.baseline_feature_keys = []
        self.one_hot_keys = []

        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.module.items():
            if k == 'time':
                continue
            if item['arc'] == 'one_hot':
                encoder = nn.Embedding(num_embeddings=item['num_embeddings'], embedding_dim=item['embedding_dim'])
                torch.nn.init.xavier_uniform_(encoder.weight)
                self.encode_modules[k] = encoder
                self.one_hot_keys.append(k)
            elif item['arc'] == 'fc':
                encoder = fc_block(item['input_dim'], item['output_dim'], activation=self.act)
                self.encode_modules[k] = encoder
            if 'scalar_context' in item.keys() and item['scalar_context']:
                self.scalar_context_keys.append(k)
            if 'baseline_feature' in item.keys() and item['baseline_feature']:
                self.baseline_feature_keys.append(k)

        self.position_array = torch.nn.Parameter(
            compute_denominator(
                torch.arange(0, self.cfg.module.time.output_dim, dtype=torch.float), self.cfg.module.time.output_dim
            ),
            requires_grad=False
        )
        self.time_embedding_dim = self.cfg.module.time.output_dim
        bo_cfg = self.cfg.module.beginning_order
        self.encode_modules['beginning_order'] = BeginningBuildOrderEncoder(bo_cfg)

    def time_encoder(self, x: Tensor):
        v = torch.zeros(size=(x.shape[0], self.time_embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v

    def forward(self, x: Dict[str, Tensor]):
        embedded_scalar = []
        scalar_context = []
        baseline_feature = []

        for key, item in self.cfg.module.items():
            assert key in x.keys(), key
            if key == 'time':
                continue
            elif item['arc'] == 'one_hot':
                # check data
                over_cross_data = x[key] >= item['num_embeddings']
                if over_cross_data.any():
                    print(key, x[key][over_cross_data])

                x[key] = x[key].clamp_(max=item['num_embeddings'] - 1)
                embedding = self.encode_modules[key](x[key].long())
                embedding = self.act(embedding)
            elif key == 'beginning_order':
                embedding = self.encode_modules[key](x[key].float(), x['bo_location'].long())
            else:
                embedding = self.encode_modules[key](x[key].float())
            embedded_scalar.append(embedding)
            if key in self.scalar_context_keys:
                scalar_context.append(embedding)
            if key in self.baseline_feature_keys:
                baseline_feature.append(embedding)
        time_embedding = self.time_encoder(x['time'])
        embedded_scalar.append(time_embedding)
        embedded_scalar = torch.cat(embedded_scalar, dim=1)
        scalar_context = torch.cat(scalar_context, dim=1)
        baseline_feature = torch.cat(baseline_feature, dim=1)

        return embedded_scalar, scalar_context, baseline_feature
