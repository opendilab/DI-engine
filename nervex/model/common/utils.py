import torch
from easydict import EasyDict
from nervex.utils import import_module, MODEL_REGISTRY


def create_model(cfg: EasyDict) -> torch.nn.Module:
    import_module(cfg.pop('import_names', []))
    return MODEL_REGISTRY.build(cfg.pop("model_type"), **cfg)
