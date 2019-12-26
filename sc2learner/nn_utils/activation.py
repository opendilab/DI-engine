import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(context_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x, context):
        gate = self.fc1(context)
        gate = torch.sigmoid(gate)
        x = gate * x
        x = self.fc2(x)
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
