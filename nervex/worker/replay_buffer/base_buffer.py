from abc import ABC, abstractmethod
import time
import copy
import os.path as osp
from threading import Thread
from typing import Union, Optional, Dict, Any, List, Tuple
from easydict import EasyDict

from nervex.utils import import_module, BUFFER_REGISTRY
from nervex.utils import LockContext, LockContextType, EasyTimer, build_logger, deep_merge_dicts


class IBuffer(ABC):

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def push(self, data: Union[list, dict], cur_collector_envstep: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, info: Dict[str, list]) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, cur_learner_iter: int) -> list:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, _state_dict: dict) -> None:
        raise NotImplementedError


def create_buffer(cfg: EasyDict, *args, **kwargs) -> IBuffer:
    import_module(cfg.get('import_names', []))
    if cfg.type == 'naive':
        kwargs.pop('tb_logger', None)
    return BUFFER_REGISTRY.build(cfg.type, cfg, *args, **kwargs, name=cfg.pop('name', 'default'))


def get_buffer_cls(cfg: EasyDict) -> type:
    import_module(cfg.get('import_names', []))
    return BUFFER_REGISTRY.get(cfg.type)
