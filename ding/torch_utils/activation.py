import torch
import torch.nn as nn


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


def get_activation(name: str):
    name = name.lower()
    if name not in NONLINEARITIES:
        raise ValueError("Unknown activation function {}".format(name))
    return NONLINEARITIES[name]
