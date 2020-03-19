import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.torch_utils import conv2d_block, fc_block, build_activation, ResBlock


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
        for i in range(cfg.resblock_num):
            self.res.append(ResBlock(dim, dim, 3, 1, 1, activation=self.act, norm_type=self.norm))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_block(dim, cfg.fc_dim, activation=self.act)

    def forward(self, x, map_size):
        '''
        Arguments:
            x: [batch_size, input_dim, H, W]
            map_size: list[list]  (y, x)
        Returns:
            output: [batch_size, fc_dim]
            map_skip: list[Tensor(batch_size, 128, H//8, W//8) x 4]
        '''
        if isinstance(x, torch.Tensor):
            return self._forward(x, map_size)
        elif isinstance(x, list):
            output = []
            map_skip = []
            for item in x:
                o, m = self._forward(item.unsqueeze(0), map_size)
                output.append(o)
                map_skip.append(m)
            output = torch.stack(output, dim=0)
            map_skip = list(zip(*map_skip))
            return output, map_skip
        else:
            raise TypeError("invalid input type: {}".format(type(x)))

    def _top_left_crop(self, data, map_size):
        ratio = int(math.pow(2, len(self.down_channels)))
        new_data = []
        for d, m in zip(data, map_size):
            h, w = m
            h, w = h // ratio, w // ratio
            new_data.append(d[..., :h, :w].unsqueeze(0))
        if len(new_data) == 1:
            new_data = new_data[0]
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
                output.append(self.gap(t))
            x = torch.cat(output, dim=0)
            del output
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x, map_skip


def test_spatial_encoder():
    class CFG:
        def __init__(self):
            self.fc_dim = 256
            self.resblock_num = 4
            self.input_dim = 40
            self.project_dim = 32
            self.downsample_type = 'conv2d'
            self.down_channels = [64, 128, 128]
            self.activation = 'relu'
            self.norm_type = 'BN'

    model = SpatialEncoder(CFG()).cuda()
    input = torch.randn(4, 40, 128, 128).cuda()
    output, map_skip = model(input)
    print(model)
    print(output.shape)
    for idx, item in enumerate(map_skip):
        print(idx, item.shape)


if __name__ == "__main__":
    test_spatial_encoder()
