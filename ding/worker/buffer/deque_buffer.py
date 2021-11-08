from typing import Any, Iterable, List, Optional, Union
from collections import defaultdict, deque
from ding.worker.buffer import Buffer, apply_middleware, BufferedData
from ding.worker.buffer.utils import fastcopy
import itertools
import random
import uuid
import logging


class DequeBuffer(Buffer):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.storage = deque(maxlen=size)

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        index = uuid.uuid1().hex
        buffered = BufferedData(data=data, index=index, meta=meta)
        self.storage.append(buffered)
        return buffered

    @apply_middleware("sample")
    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
    ) -> List[BufferedData]:
        storage = self.storage
        if sample_range:
            storage = list(itertools.islice(self.storage, sample_range.start, sample_range.stop, sample_range.step))
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")
        if not size:
            size = len(indices)

        value_error = None
        sampled_data = []
        if indices:
            indices_set = set(indices)
            hashed_data = filter(lambda item: item.index in indices_set, self.storage)
            hashed_data = map(lambda item: (item.index, item), hashed_data)
            hashed_data = dict(hashed_data)
            # Re-sample and return in indices order
            sampled_data = [hashed_data[index] for index in indices]
        else:
            if replace:
                sampled_data = random.choices(storage, k=size)
            else:
                try:
                    sampled_data = random.sample(storage, k=size)
                except ValueError as e:
                    value_error = e

        if value_error or len(sampled_data) != size:
            if ignore_insufficient:
                logging.warning(
                    "Sample operation is ignored due to data insufficient, current buffer count is {} while sample size is {}"
                    .format(self.count(), size)
                )
            else:
                raise ValueError("There are less than {} data in buffer({})".format(size, self.count()))

        sampled_data = self._independence(sampled_data)

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

    def _independence(self, buffered_samples: List[BufferedData]) -> List[BufferedData]:
        """
        Overview:
            Make sure that each record is different from each other, but remember that this function
            is different from clone_object. You may change the data in the buffer by modifying a record.
        """
        occurred = defaultdict(int)
        for i, buffered in enumerate(buffered_samples):
            occurred[buffered.index] += 1
            if occurred[buffered.index] > 1:
                buffered_samples[i] = fastcopy.copy(buffered)
        return buffered_samples

    def __iter__(self) -> deque:
        return iter(self.storage)

    def __copy__(self) -> "DequeBuffer":
        buffer = type(self)(size=self.storage.maxlen)
        buffer.storage = self.storage
        return buffer
