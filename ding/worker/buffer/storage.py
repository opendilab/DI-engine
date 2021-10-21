from abc import abstractmethod
from typing import Any, List


class Storage:

    @abstractmethod
    def append(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, indices: List[int]) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, size: int, replace: bool = False) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
