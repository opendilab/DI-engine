from typing import Any, List, Optional
from collections import deque
from operator import itemgetter

from numpy import delete
from ding.worker.buffer import Storage
import itertools
import random


class MemoryStorage(Storage):

    def __init__(self, maxlen: int) -> None:
        self.storage = deque(maxlen=maxlen)

    def append(self, data: Any) -> None:
        self.storage.append(data)

    def get(self, indices: List[int]) -> List[Any]:
        return itemgetter(*indices)(self.storage)

    def sample(self, size: int, replace: bool = False, range: Optional[slice] = None) -> List[Any]:
        storage = self.storage
        if range:
            storage = list(itertools.islice(self.storage, range.start, range.stop, range.step))
        if replace:
            return random.choices(storage, k=size)
        else:
            return random.sample(storage, k=size)

    def update(self, index: str, data: Any, extra: Optional[Any] = None) -> None:
        pass

    def delete(self, index: str) -> None:
        pass

    def count(self) -> int:
        return len(self.storage)

    def clear(self) -> None:
        self.storage.clear()
