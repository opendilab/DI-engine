from abc import abstractmethod
from typing import Any

from .env_element import EnvElement, IEnvElement, EnvElementInfo
from ..env.base_env import BaseEnv


class IEnvElementRunner(IEnvElement):

    @abstractmethod
    def get(self, engine: BaseEnv) -> Any:
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError


class EnvElementRunner(IEnvElementRunner):

    def __init__(self, *args, **kwargs) -> None:
        self._init(*args, **kwargs)
        self._check()

    @abstractmethod
    def _init(self, *args, **kwargs) -> None:
        # set self._core and other state variable
        raise NotImplementedError

    def _check(self) -> None:
        flag = [hasattr(self, '_core'), isinstance(self._core, EnvElement)]
        assert all(flag), flag

    def __repr__(self) -> str:
        return repr(self._core)

    @property
    def info(self) -> 'EnvElementInfo':
        return self._core.info
