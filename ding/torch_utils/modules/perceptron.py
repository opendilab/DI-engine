import torch
from torch import nn
from ding.torch_utils.activation import get_activation


class multilayer_perceptron(nn.Module):

    def __init__(self, cfg):
        super(multilayer_perceptron, self).__init__()

        self.model = nn.Sequential()

        for i in range(len(cfg.hidden_sizes) - 1):
            self.model.add_module('linear' + str(i), nn.Linear(cfg.hidden_sizes[i], cfg.hidden_sizes[i + 1]))

            if isinstance(cfg.activation, list):
                self.model.add_module('activation' + str(i), get_activation(cfg.activation[i]))
            else:
                self.model.add_module('activation' + str(i), get_activation(cfg.activation))
            if hasattr(cfg, "dropout") and cfg.dropout > 0:
                self.model.add_module('dropout', nn.Dropout(cfg.dropout))
            if hasattr(cfg, "layernorm") and cfg.layernorm:
                self.model.add_module('layernorm', nn.LayerNorm(cfg.hidden_sizes[i]))

        self.model.add_module(
            'linear' + str(len(cfg.hidden_sizes) - 1), nn.Linear(cfg.hidden_sizes[-1], cfg.output_size)
        )

        if hasattr(cfg, 'final_activation'):
            self.model.add_module('final_activation', get_activation(cfg.final_activation))

        if hasattr(cfg, 'scale'):
            self.scale = nn.Parameter(torch.tensor(cfg.scale), requires_grad=False)
        else:
            self.scale = 1.0

        if hasattr(cfg, 'offset'):
            self.offset = nn.Parameter(torch.tensor(cfg.offset), requires_grad=False)
        else:
            self.offset = 0.0

        # shrink the weight of linear layer 'linear'+str(len(cfg.hidden_sizes) to it's origin 0.01
        if hasattr(cfg, 'shrink'):
            if hasattr(cfg, 'final_activation'):
                self.model[-2].weight.data.normal_(0, cfg.shrink)
                self.model[-2].bias.data.normal_(0, cfg.shrink)
            else:
                self.model[-1].weight.data.normal_(0, cfg.shrink)
                self.model[-1].bias.data.normal_(0, cfg.shrink)

    def forward(self, x):
        return self.scale * self.model(x) + self.offset
