from abc import abstractmethod
from typing import Any, List

from ding.worker.buffer.storage import Storage


class Buffer:

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    @abstractmethod
    def push(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, size: int) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
