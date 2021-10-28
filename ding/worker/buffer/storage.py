from abc import abstractmethod
from typing import Any, List, Optional


class Storage:

    @abstractmethod
    def append(self, data: Any, index: Optional[str] = None, extra: Optional[dict] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, indices: List[int]) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def sample(self, size: int, replace: bool = False, range: Optional[slice] = None) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def update(self, index: str, data: Any, extra: Optional[Any] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, index: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
