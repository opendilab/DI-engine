import torch.nn as nn


def build_activation(activation):
    act_func = {
        'relu': nn.ReLU(inplace=True),
    }
    if activation in act_func.keys():
        return act_func[activation]
    else:
        raise KeyError("invalid key for activation: {}".format(activation))
