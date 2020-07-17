from abc import ABC, abstractmethod
from typing import Any
from .env_element import EnvElement
from ..env.base_env import BaseEnv


class EnvElementRunner(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._init(*args, **kwargs)
        self._check()

    @abstractmethod
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        raise NotImplementedError

    @abstractmethod
    def get(self, engine: BaseEnv) -> Any:
        raise NotImplementedError

    def _check(self) -> None:
        flag = [hasattr(self, '_core'), isinstance(self._core, EnvElement)]
        assert all(flag), flag

    def __repr__(self) -> str:
        return repr(self._core)

    @property
    def info(self) -> 'EnvElement.info_template':
        return self._core.info
