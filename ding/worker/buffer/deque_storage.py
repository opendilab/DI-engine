import enum
from typing import Any, List, Optional, Tuple, Union
from collections import deque
from ding.worker.buffer import Storage
import itertools
import random
import uuid


class DequeStorage(Storage):

    def __init__(self, maxlen: int) -> None:
        self.storage = deque(maxlen=maxlen)

    def append(self, data: Any, meta: Optional[dict] = None) -> None:
        index = uuid.uuid1().hex
        self.storage.append((data, index, meta))

    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            range: Optional[slice] = None,
            return_index: bool = False,
            return_meta: bool = False
    ) -> List[Union[Any, Tuple[Any, str], Tuple[Any, dict], Tuple[Any, str, dict]]]:
        storage = self.storage
        if range:
            storage = list(itertools.islice(self.storage, range.start, range.stop, range.step))
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")

        if indices:
            sampled_data = filter(lambda item: item[1] in indices, self.storage)
        else:
            sampled_data = random.choices(storage, k=size) if replace else random.sample(storage, k=size)

        if return_index and not return_meta:
            sampled_data = list(map(lambda item: (item[0], item[1]), sampled_data))
        elif not return_index and return_meta:
            sampled_data = list(map(lambda item: (item[0], item[2]), sampled_data))
        elif not return_index and not return_meta:
            sampled_data = list(map(lambda item: item[0], sampled_data))

        return sampled_data

    def update(self, index: str, data: Any, meta: Optional[Any] = None) -> bool:
        for i, (_, _index, _) in enumerate(self.storage):
            if _index == index:
                self.storage[i] = (data, _index, meta)
                return True
        return False

    def delete(self, index: str) -> bool:
        for i, (_, _index, _) in enumerate(self.storage):
            if _index == index:
                del self.storage[i]
                return True
        return False

    def count(self) -> int:
        return len(self.storage)

    def clear(self) -> None:
        self.storage.clear()

    def __iter__(self) -> deque:
        return iter(self.storage)
