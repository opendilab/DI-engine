import torch
from easydict import EasyDict
from ding.utils import import_module, MODEL_REGISTRY


def create_model(cfg: EasyDict) -> torch.nn.Module:
    r"""
    Overview:
        Creat model given config dictionary
    Arguments:
        - cfg: (:obj:`dict`):
            The trainning configuration, the key ``import_name`` is
            used to import module, and they key ``model_type`` is used to build model.
    Returns:
        - (:obj:`torch.nn.Module`) The corresponding model.
    """
    import_module(cfg.pop('import_names', []))
    return MODEL_REGISTRY.build(cfg.pop("model_type"), **cfg)
