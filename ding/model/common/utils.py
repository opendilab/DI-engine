import torch
import torch.nn as nn
from easydict import EasyDict
from ding.utils import import_module, MODEL_REGISTRY

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
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

def get_activation(name:str):
    name=name.lower()
    if name not in NONLINEARITIES:
        raise ValueError("Unknown activation function {}".format(name))
    return NONLINEARITIES[name]

def create_model(cfg: EasyDict) -> torch.nn.Module:
    """
    Overview:
        Create a model given a config dictionary.
    Arguments:
        - cfg: (:obj:`dict`): Training configuration. The key ``import_name`` is \
            used to import modules, and they key ``type`` is used to build the model.
    Returns:
        - (:obj:`torch.nn.Module`) Training configuration corresponding model.
    """
    import_module(cfg.pop('import_names', []))
    # must use pop
    return MODEL_REGISTRY.build(cfg.pop("type"), **cfg)
