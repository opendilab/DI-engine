import copy
import torch
from easydict import EasyDict
from ding.utils import import_module, MODEL_REGISTRY


def create_model(cfg: EasyDict) -> torch.nn.Module:
    """
    Overview:
        Create a neural network model according to the given EasyDict-type ``cfg``.
    Arguments:
        - cfg: (:obj:`EasyDict`): User's model config. The key ``import_name`` is \
            used to import modules, and they key ``type`` is used to indicate the model.
    Returns:
        - (:obj:`torch.nn.Module`): The created neural network model.

    .. tip::
        This method will not modify the ``cfg`` , it will deepcopy the ``cfg`` and then modify it.
    """
    cfg = copy.deepcopy(cfg)
    import_module(cfg.pop('import_names', []))
    # here we must use the pop opeartion to ensure compatibility
    return MODEL_REGISTRY.build(cfg.pop("type"), **cfg)
