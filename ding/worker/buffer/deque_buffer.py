import enum
from typing import Any, Iterable, List, Optional, Tuple, Union
from collections import deque
from ding.worker.buffer import Buffer, apply_middleware, BufferedData
import itertools
import random
import uuid


class DequeBuffer(Buffer):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.storage = deque(maxlen=size)

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> None:
        index = uuid.uuid1().hex
        self.storage.append(BufferedData(data=data, index=index, meta=meta))

    @apply_middleware("sample")
    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            range: Optional[slice] = None
    ) -> List[BufferedData]:
        storage = self.storage
        if range:
            storage = list(itertools.islice(self.storage, range.start, range.stop, range.step))
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")

        if indices:
            sampled_data = filter(lambda item: item.index in indices, self.storage)
        else:
            sampled_data = random.choices(storage, k=size) if replace else random.sample(storage, k=size)
        return sampled_data

    @apply_middleware("update")
    def update(self, index: str, data: Any, meta: Optional[Any] = None) -> bool:
        for item in self.storage:
            if item.index == index:
                item.data = data
                item.meta = meta
                return True
        return False

    @apply_middleware("delete")
    def delete(self, indices: Union[str, Iterable[str]]) -> None:
        if isinstance(indices, str):
            indices = [indices]
        for i in indices:
            for index, item in enumerate(self.storage):
                if item.index == i:
                    del self.storage[index]
                    break

    def count(self) -> int:
        return len(self.storage)

    @apply_middleware("clear")
    def clear(self) -> None:
        self.storage.clear()

    def __iter__(self) -> deque:
        return iter(self.storage)

    def __copy__(self) -> "DequeBuffer":
        buffer = type(self)(size=self.storage.maxlen)
        buffer.storage = self.storage
        return buffer
