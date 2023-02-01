from abc import ABC, abstractmethod
from typing import Any


class Storage(ABC):

    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def save(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError
