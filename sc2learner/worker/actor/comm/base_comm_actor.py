from abc import ABC, abstractmethod
from typing import Any


class BaseCommActor(ABC):
    @abstractmethod
    def get_agent_update_info(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def send_traj_metadata(self, metadata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_traj_stepdata(self, stepdata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_finish_job(self, finish_info: dict) -> None:
        raise NotImplementedError


class ActorCommMetalass(type):
    def __new__(cls, name, bases, attrs):
        attrs['__init__'] = cls.enable_comm_helper(attrs['__init__'])
        return type.__new__(cls, name, bases, attrs)

    @classmethod
    def enable_comm_helper(cls, fn):
        def wrapper(*args, **kwargs):
            if 'comm_cfg' in kwargs.keys():
                comm_cfg = kwargs.pop('comm_cfg')
            return fn(*args, **kwargs)

        return wrapper


class SingleMachineActor(ABC):
    # TODO single matchine actor for some micro envs
    pass
