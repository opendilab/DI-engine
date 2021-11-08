from typing import Any, Iterable, List, Optional, Union
from collections import deque
import itertools
import random
import uuid
import logging
from ding.worker.buffer import Buffer, apply_middleware, BufferedData


class DequeBuffer(Buffer):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.storage = deque(maxlen=size)

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> str:
        index = uuid.uuid1().hex
        self.storage.append(BufferedData(data=data, index=index, meta=meta))
        return index

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
            sampled_data = list(filter(lambda item: item.index in indices, self.storage))
            # for the same indices
            if len(indices) != len(set(indices)):
                sampled_data_no_same = sampled_data
                sampled_data = [sampled_data_no_same[0]]
                j = 0
                for i in range(1, len(indices)):
                    if indices[i - 1] == indices[i]:
                        sampled_data.append(copy.deepcopy(sampled_data_no_same[j]))
                    else:
                        sampled_data.append(sampled_data_no_same[j])
                        j += 1
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
                if value_error:
                    raise ValueError("Some errors in sample operation") from value_error
                else:
                    raise ValueError("There are less than {} data in buffer({})".format(size, self.count()))

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
