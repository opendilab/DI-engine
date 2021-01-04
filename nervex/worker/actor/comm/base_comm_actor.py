import threading
from abc import ABC, abstractmethod
from typing import Any
from easydict import EasyDict

from nervex.policy import create_policy
from nervex.utils import get_task_uid, import_module
from ..base_parallel_actor import create_actor, BaseActor


class BaseCommActor(ABC):

    def __init__(self, cfg):
        self._cfg = cfg
        self._end_flag = True
        self._actor_uid = get_task_uid()

    @abstractmethod
    def get_policy_update_info(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def send_metadata(self, metadata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_stepdata(self, stepdata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_finish_info(self, path: str, finish_info: Any) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._end_flag = False

    def close(self) -> None:
        self._end_flag = True

    @property
    def actor_uid(self) -> str:
        return self._actor_uid

    def _create_actor(self, task_info: dict) -> BaseActor:
        actor_cfg = EasyDict(task_info['actor_cfg'])
        actor = create_actor(actor_cfg)
        for item in ['send_metadata', 'send_stepdata', 'get_policy_update_info', 'send_finish_info']:
            setattr(actor, item, getattr(self, item))
        actor.policy = create_policy(task_info['policy'], enable_field=['collect']).collect_mode
        return actor


comm_map = {}


def register_comm_actor(name: str, actor_type: type) -> None:
    assert isinstance(name, str)
    assert issubclass(actor_type, BaseCommActor)
    comm_map[name] = actor_type


def create_comm_actor(cfg: dict) -> BaseCommActor:
    cfg = EasyDict(cfg)
    import_module(cfg.import_names)
    comm_actor_type = cfg.comm_actor_type
    if comm_actor_type not in comm_map.keys():
        raise KeyError("not support comm actor type: {}".format(comm_actor_type))
    else:
        return comm_map[comm_actor_type](cfg)
