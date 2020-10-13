from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List


class BaseEnv(ABC):
    """
    Overview: basic environment class
    Interface: __init__
    Property: timestep
    """
    timestep = namedtuple('BaseEnvTimestep', ['obs', 'act', 'reward', 'done', 'info'])
    info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> 'BaseEnv.info':
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def pack(self, timesteps: List['BaseEnv.timestep'] = None, obs: Any = None) -> 'BaseEnv.timestep':
        raise NotImplementedError

    @abstractmethod
    def unpack(self, action: Any) -> List[Any]:
        raise NotImplementedError
