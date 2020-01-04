import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import conv2d_block, fc_block, build_activation, ResBlock


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super(SpatialEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.norm = cfg.norm_type
        self.project = conv2d_block(cfg.input_dim, cfg.project_dim, 1, 1, 0, activation=self.act, norm_type=self.norm)
        down_layers = []
        dims = [cfg.project_dim] + cfg.down_channels
        for i in range(len(cfg.down_channels)):
            if cfg.downsample_type == 'conv2d':
                down_layers.append(conv2d_block(dims[i], dims[i+1], 4, 2, 1, activation=self.act, norm_type=self.norm))
            elif cfg.downsample_type == 'avgpool':
                down_layers.append(nn.AvgPool2d(2, 2))
                down_layers.append(conv2d_block(dims[i], dims[i+1], 3, 1, 1, activation=self.act, norm_type=self.norm))
            else:
                raise KeyError("invalid downsample module type :{}".format(type(cfg.downsample_type)))
        self.downsample = nn.Sequential(*down_layers)
        self.res = nn.ModuleList()
        dim = dims[-1]
        for i in range(cfg.resblock_num):
            self.res.append(ResBlock(dim, dim, 3, 1, 1, activation=self.act, norm_type=self.norm))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_block(dim, cfg.fc_dim, activation=self.act)

    def forward(self, x):
        '''
        Input:
            x: [batch_size, input_dim, 128, 128]
        Output:
            x: [batch_size, fc_dim]
            map_skip: list[Tensor(batch_size, 128, 16, 16) x 4]
        '''
        x = self.project(x)
        x = self.downsample(x)
        map_skip = []
        for block in self.res:
            x = block(x)
            map_skip.append(x)
        x = self.gap(x)
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x, map_skip


def transform_spatial_data():
    template = {
        {'key': 'camera', 'other': 'one-hot 2 value'},
        {'key': 'scattered_entities', 'other': '32 channel float'},
        {'key': 'height_map', 'other': 'float height_map/255'},
        {'key': 'visibility', 'other': 'one-hot 4 value'},
        {'key': 'creep', 'other': 'one-hot 2 value'},
        {'key': 'entity_owners', 'other': 'one-hot 5 value'},
        {'key': 'alerts', 'other': 'one-hot 2 value'},
        {'key': 'pathable', 'other': 'one-hot 2 value'},
        {'key': 'buildable', 'other': 'one-hot 2 value'},
    }


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
