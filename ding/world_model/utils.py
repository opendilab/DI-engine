from easydict import EasyDict
from typing import Callable
from ding.utils import import_module, WORLD_MODEL_REGISTRY


def get_world_model_cls(cfg):
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.get(cfg.type)


def create_world_model(cfg, *args, **kwargs):
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.build(cfg.type, cfg, *args, **kwargs)


def get_rollout_length_scheduler(cfg: EasyDict) -> Callable[[int], int]:
    """
    Overview:
        Get the rollout length scheduler that adapts rollout length based\
        on the current environment steps.
    Returns:
        - scheduler (:obj:`Callble`): The function that takes envstep and\
          return the current rollout length.
    """
    if cfg.type == 'linear':
        x0 = cfg.rollout_start_step
        x1 = cfg.rollout_end_step
        y0 = cfg.rollout_length_min
        y1 = cfg.rollout_length_max
        w = (y1 - y0) / (x1 - x0)
        b = y0
        return lambda x: int(min(max(w * (x - x0) + b, y0), y1))
    elif cfg.type == 'constant':
        return lambda x: cfg.rollout_length
    else:
        raise KeyError("not implemented key: {}".format(cfg.type))
