import torch
from easydict import EasyDict
from ding.utils import import_module, MODEL_REGISTRY


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
