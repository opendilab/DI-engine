import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim, input_type='fc'):
        super(GLU, self).__init__()
        assert(input_type in ['fc', 'conv2d'])
        if input_type == 'fc':
            self.layer1 = nn.Linear(context_dim, input_dim)
            self.layer2 = nn.Linear(input_dim, output_dim)
        elif input_type == 'conv2d':
            self.layer1 = nn.Conv2d(context_dim, input_dim, 1, 1, 0)
            self.layer2 = nn.Conv2d(input_dim, output_dim, 1, 1, 0)

    def forward(self, x, context):
        gate = self.layer1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.layer2(x)
        return x


def build_activation(activation):
    act_func = {
        'relu': nn.ReLU(inplace=True),
        'glu': GLU
    }
    if activation in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))
