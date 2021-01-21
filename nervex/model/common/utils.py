import torch
from easydict import EasyDict
from nervex.utils import import_module

model_mapping = {}


def register_model(name: str, model: type) -> None:
    assert isinstance(name, str)
    model_mapping[name] = model


def create_model(cfg: EasyDict) -> torch.nn.Module:
    import_module(cfg.pop('import_names'))
    model_type = cfg.pop('model_type')
    if model_type not in model_mapping.keys():
        raise KeyError("not support model type: {}".format(model_type))
    else:
        return model_mapping[model_type](**cfg)
