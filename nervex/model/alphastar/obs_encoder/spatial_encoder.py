import math
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from nervex.torch_utils import conv2d_block, fc_block, build_activation, ResBlock, same_shape


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super(SpatialEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.norm = cfg.norm_type
        self.project = conv2d_block(cfg.input_dim, cfg.project_dim, 1, 1, 0, activation=self.act, norm_type=self.norm)
        down_layers = []
        dims = [cfg.project_dim] + cfg.down_channels
        self.down_channels = cfg.down_channels
        for i in range(len(self.down_channels)):
            if cfg.downsample_type == 'conv2d':
                down_layers.append(
                    conv2d_block(dims[i], dims[i + 1], 4, 2, 1, activation=self.act, norm_type=self.norm)
                )
            elif cfg.downsample_type == 'avgpool':
                down_layers.append(nn.AvgPool2d(2, 2))
                down_layers.append(
                    conv2d_block(dims[i], dims[i + 1], 3, 1, 1, activation=self.act, norm_type=self.norm)
                )
            else:
                raise KeyError("invalid downsample module type :{}".format(type(cfg.downsample_type)))
        self.downsample = nn.Sequential(*down_layers)
        self.res = nn.ModuleList()
        dim = dims[-1]
        self.resblock_num = cfg.resblock_num
        for i in range(cfg.resblock_num):
            self.res.append(ResBlock(dim, dim, 3, 1, 1, activation=self.act, norm_type=self.norm))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_block(dim, cfg.fc_dim, activation=self.act)

    def forward(self, x, map_size):
        '''
        Arguments:
            x: [batch_size, input_dim, H, W]
            map_size: list[len=batch_size]->element: list[len=2] (y, x)
        Returns:
            output: [batch_size, fc_dim]
            map_skip: list[len=resblock_num]->element: list[len=batch_size]->
                element: torch.Tensor(down_channels[-1], H//8, W//8) x 4]
        '''
        if isinstance(x, torch.Tensor):
            return self._forward(x, map_size)
        elif isinstance(x, Sequence):
            output = []
            map_skip = []
            for item, m in zip(x, map_size):
                o, m = self._forward(item.unsqueeze(0), [m])
                output.append(o)
                map_skip.append(m)
            output = torch.cat(output, dim=0)
            map_skip = [[map_skip[j][i][0] for j in range(len(map_skip))] for i in range(self.resblock_num)]
            return output, map_skip
        else:
            raise TypeError("invalid input type: {}".format(type(x)))

    def _top_left_crop(self, data, map_size):
        ratio = int(math.pow(2, len(self.down_channels)))
        new_data = []
        for d, m in zip(data, map_size):
            h, w = m
            h, w = h // ratio, w // ratio
            new_data.append(d[..., :h, :w])
        if same_shape(new_data):
            new_data = torch.stack(new_data, dim=0)
        return new_data

    def _forward(self, x, map_size):
        x = self.project(x)
        x = self.downsample(x)
        map_skip = []
        for block in self.res:
            x = block(x)
            map_skip.append(self._top_left_crop(x, map_size))
        x = self._top_left_crop(x, map_size)
        if isinstance(x, torch.Tensor):
            x = self.gap(x)
        elif isinstance(x, list):
            output = []
            for idx, t in enumerate(x):
                output.append(self.gap(t.unsqueeze(0)))
            x = torch.cat(output, dim=0)
            del output
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x, map_skip
