from typing import Any, List
from collections import deque
from operator import itemgetter
from ding.worker.buffer import Storage
import random
import numpy as np


class MemoryStorage(Storage):

    def __init__(self, maxlen: int) -> None:
        self.storage = deque(maxlen=maxlen)

    def append(self, data: Any) -> None:
        self.storage.append(data)

    def get(self, indices: List[int]) -> List[Any]:
        return itemgetter(*indices)(self.storage)

    def sample(self, size: int, replace: bool = False) -> List[Any]:
        return np.random.choice(self.storage, size, replace=replace)

    def count(self) -> int:
        return len(self.storage)

    def clear(self) -> None:
        self.storage.clear()
