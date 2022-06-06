import math
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.torch_utils import conv2d_block, fc_block, build_activation, ResBlock, same_shape


class SpatialEncoder(nn.Module):
    __constants__ = ['head_type']

    def __init__(self, cfg):
        super(SpatialEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.encoder.obs_encoder.spatial_encoder
        self.act = build_activation(self.cfg.activation)
        if self.cfg.norm_type == 'none':
            self.norm = None
        else:
            self.norm = self.cfg.norm_type
        self.project = conv2d_block(
            self.cfg.input_dim, self.cfg.project_dim, 1, 1, 0, activation=self.act, norm_type=self.norm
        )
        dims = [self.cfg.project_dim] + self.cfg.down_channels
        self.down_channels = self.cfg.down_channels
        self.encode_modules = nn.ModuleDict()
        self.downsample = nn.ModuleList()
        for k, item in self.cfg.module.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = nn.Embedding.from_pretrained(
                    torch.eye(item['num_embeddings']), freeze=True, padding_idx=None
                )
        for i in range(len(self.down_channels)):
            if self.cfg.downsample_type == 'conv2d':
                self.downsample.append(
                    conv2d_block(dims[i], dims[i + 1], 4, 2, 1, activation=self.act, norm_type=self.norm)
                )
            elif self.cfg.downsample_type in ['avgpool', 'maxpool']:
                self.downsample.append(
                    conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=self.norm)
                )
            else:
                raise KeyError("invalid downsample module type :{}".format(type(self.cfg.downsample_type)))
        self.res = nn.ModuleList()
        self.head_type = self.cfg.get('head_type', 'pool')
        dim = dims[-1]
        self.resblock_num = self.cfg.resblock_num
        for i in range(self.cfg.resblock_num):
            self.res.append(ResBlock(dim, self.act, norm_type=self.norm))
        if self.head_type == 'fc':
            self.fc = fc_block(
                dim * self.whole_cfg.model.spatial_y // 8 * self.whole_cfg.model.spatial_x // 8,
                self.cfg.fc_dim,
                activation=self.act
            )
        else:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = fc_block(dim, self.cfg.fc_dim, activation=self.act)

    def forward(self, x, scatter_map):
        embeddings = []
        for k, item in self.cfg.module.items():
            if item['arc'] == 'one_hot':
                embedding = self.encode_modules[k](x[k].long())
                embedding = embedding.permute(0, 3, 1, 2)
                embeddings.append(embedding)
            elif item['arc'] == 'other':
                assert k == 'height_map'
                embeddings.append(x[k].unsqueeze(dim=1).float() / 256)
            elif item['arc'] == 'scatter':
                bs, shape_x, shape_y = x[k].shape[0], self.whole_cfg.model.spatial_x, self.whole_cfg.model.spatial_y
                embedding = torch.zeros(bs * shape_y * shape_x, device=x[k].device)
                bias = torch.arange(bs, device=x[k].device).unsqueeze(dim=1) * shape_y * shape_x
                x[k] = x[k] + bias
                x[k] = x[k].view(-1)
                embedding[x[k].long()] = 1.
                embedding = embedding.view(bs, 1, shape_y, shape_x)
                embeddings.append(embedding)
        embeddings.append(scatter_map)
        x = torch.cat(embeddings, dim=1)
        x = self.project(x)
        map_skip = []
        for i in range(len(self.downsample)):
            map_skip.append(x)
            if self.cfg.downsample_type == 'avgpool':
                x = torch.nn.functional.avg_pool2d(x, 2, 2)
            elif self.cfg.downsample_type == 'maxpool':
                x = torch.nn.functional.max_pool2d(x, 2, 2)
            x = self.downsample[i](x)
        for block in self.res:
            map_skip.append(x)
            x = block(x)
        if isinstance(x, torch.Tensor):
            if self.head_type != 'fc':
                x = self.gap(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, map_skip
