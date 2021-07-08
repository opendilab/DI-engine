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
            used to import module, and they key ``type`` is used to build model.
    Returns:
        - (:obj:`torch.nn.Module`) The corresponding model.
    """
    import_module(cfg.pop('import_names', []))
    # must use pop
    return MODEL_REGISTRY.build(cfg.pop("type"), **cfg)
