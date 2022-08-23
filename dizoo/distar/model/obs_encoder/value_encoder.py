import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import fc_block, build_activation, conv2d_block, ResBlock, same_shape, sequence_mask, \
    Transformer, scatter_connection_v2
from dizoo.distar.envs import BEGIN_ACTIONS
from .spatial_encoder import SpatialEncoder
from .scalar_encoder import BeginningBuildOrderEncoder


class ValueEncoder(nn.Module):

    def __init__(self, cfg):
        super(ValueEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = cfg.encoder
        self.act = build_activation('relu', inplace=True)
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'fc':
                self.encode_modules[k] = fc_block(item['input_dim'], item['output_dim'], activation=self.act)
            elif item['arc'] == 'one_hot':
                self.encode_modules[k] = nn.Embedding(
                    num_embeddings=item['num_embeddings'], embedding_dim=item['embedding_dim']
                )

        bo_cfg = self.cfg.modules.beginning_order
        self.encode_modules['beginning_order'] = BeginningBuildOrderEncoder(bo_cfg)
        self.scatter_project = fc_block(
            self.cfg.scatter.scatter_input_dim, self.cfg.scatter.scatter_dim, activation=self.act
        )
        self.scatter_type = self.cfg.scatter.scatter_type
        self.scatter_dim = self.cfg.scatter.scatter_dim

        self.project = conv2d_block(
            self.cfg.spatial.input_dim, self.cfg.spatial.project_dim, 1, 1, 0, activation=self.act
        )
        down_layers = []
        dims = [self.cfg.spatial.project_dim] + self.cfg.spatial.down_channels
        for i in range(len(self.cfg.spatial.down_channels)):
            down_layers.append(nn.MaxPool2d(2, 2))
            down_layers.append(conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act))
        self.downsample = nn.Sequential(*down_layers)
        dim = dims[-1]
        self.resblock_num = self.cfg.spatial.resblock_num
        self.res = nn.ModuleList()
        for i in range(self.resblock_num):
            self.res.append(ResBlock(dim, self.act, norm_type=None))
        self.spatial_fc = fc_block(
            dim * self.whole_cfg.spatial_y // 8 * self.whole_cfg.spatial_x // 8,
            self.cfg.spatial.spatial_fc_dim,
            activation=self.act
        )

    def forward(self, x):
        spatial_embedding = []
        fc_embedding = []
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'fc':
                embedding = self.encode_modules[k](x[k].float())
                fc_embedding.append(embedding)
            if item['arc'] == 'one_hot':
                embedding = self.encode_modules[k](x[k].long())
                spatial_embedding.append(embedding)

        bo_embedding = self.encode_modules['beginning_order'](x['beginning_order'].float(), x['bo_location'].long())
        fc_embedding = torch.cat(fc_embedding, dim=-1)
        spatial_embedding = torch.cat(spatial_embedding, dim=-1)
        project_embedding = self.scatter_project(spatial_embedding)
        unit_mask = sequence_mask(x['total_unit_count'], max_len=project_embedding.shape[1])
        project_embedding = project_embedding * unit_mask.unsqueeze(dim=2)
        entity_location = torch.cat([x['unit_x'].unsqueeze(dim=-1), x['unit_y'].unsqueeze(dim=-1)], dim=-1)
        b, c, h, w = x['own_units_spatial'].shape
        scatter_map = scatter_connection_v2(
            (b, h, w), project_embedding, entity_location, self.scatter_dim, self.scatter_type
        )
        spatial_x = torch.cat([scatter_map, x['own_units_spatial'].float(), x['enemy_units_spatial'].float()], dim=1)
        spatial_x = self.project(spatial_x)
        spatial_x = self.downsample(spatial_x)
        for i in range(self.resblock_num):
            spatial_x = self.res[i](spatial_x)
        spatial_x = self.spatial_fc(spatial_x.view(spatial_x.shape[0], -1))
        x = torch.cat([fc_embedding, spatial_x, bo_embedding], dim=-1)
        return x
