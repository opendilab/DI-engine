from typing import Any, Iterable, List, Optional, Tuple, Union
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
        # Meta index is a dict which use deque as values
        self.meta_index = {}

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        return self._push(data, meta)

    @apply_middleware("sample")
    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: str = None,
            rolling_window: int = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        storage = self.storage
        if sample_range:
            storage = list(itertools.islice(self.storage, sample_range.start, sample_range.stop, sample_range.step))

        # Size and indices
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")
        if not size:
            size = len(indices)
        # Indices and groupby
        assert not (indices and groupby), "Cannot use groupby and indicex at the same time."
        # Groupby and rolling_window
        assert not (groupby and rolling_window), "Cannot use groupby and rolling_window at the same time."
        assert not (indices and rolling_window), "Cannot use indices and rolling_window at the same time."

        value_error = None
        sampled_data = []
        if indices:
            indices_set = set(indices)
            hashed_data = filter(lambda item: item.index in indices_set, storage)
            hashed_data = map(lambda item: (item.index, item), hashed_data)
            hashed_data = dict(hashed_data)
            # Re-sample and return in indices order
            sampled_data = [hashed_data[index] for index in indices]
        elif groupby:
            sampled_data = self._sample_by_group(size=size, groupby=groupby, replace=replace, storage=storage)
        elif rolling_window:
            sampled_data = self._sample_by_rolling_window(
                size=size, replace=replace, rolling_window=rolling_window, storage=storage
            )
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

    def import_data(self, data_with_meta: List[Tuple[Any, dict]]) -> None:
        for data, meta in data_with_meta:
            self._push(data, meta)

    def export_data(self) -> List[BufferedData]:
        return list(self.storage)

    def _push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        index = uuid.uuid1().hex
        if meta is None:
            meta = {}
        buffered = BufferedData(data=data, index=index, meta=meta)
        self.storage.append(buffered)
        # Add meta index
        for key in self.meta_index:
            self.meta_index[key].append(meta[key] if key in meta else None)

        return buffered

    def _independence(
        self, buffered_samples: Union[List[BufferedData], List[List[BufferedData]]]
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            Make sure that each record is different from each other, but remember that this function
            is different from clone_object. You may change the data in the buffer by modifying a record.
        Arguments:
            - buffered_samples (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`) Sampled data,
                can be nested if groupby or rolling_window has been set.
        """
        if len(buffered_samples) == 0:
            return buffered_samples
        occurred = defaultdict(int)

        for i, buffered in enumerate(buffered_samples):
            if isinstance(buffered, list):
                sampled_list = buffered
                # Loop over nested samples
                for j, buffered in enumerate(sampled_list):
                    occurred[buffered.index] += 1
                    if occurred[buffered.index] > 1:
                        sampled_list[j] = fastcopy.copy(buffered)
            elif isinstance(buffered, BufferedData):
                occurred[buffered.index] += 1
                if occurred[buffered.index] > 1:
                    buffered_samples[i] = fastcopy.copy(buffered)
            else:
                raise Exception("Get unexpected buffered type {}".format(type(buffered)))
        return buffered_samples

    def _sample_by_group(self,
                         size: int,
                         groupby: str,
                         replace: bool = False,
                         storage: deque = None) -> List[List[BufferedData]]:
        if storage is None:
            storage = self.storage
        if groupby not in self.meta_index:
            self._create_index(groupby)
        meta_indices = list(set(self.meta_index[groupby]))
        sampled_groups = []
        if replace:
            sampled_groups = random.choices(meta_indices, k=size)
        else:
            try:
                sampled_groups = random.sample(meta_indices, k=size)
            except ValueError as e:
                pass
        sampled_data = defaultdict(list)
        for buffered in storage:
            meta_value = buffered.meta[groupby] if groupby in buffered.meta else None
            if meta_value in sampled_groups:
                sampled_data[buffered.meta[groupby]].append(buffered)
        return list(sampled_data.values())

    def _sample_by_rolling_window(
            self,
            size: Optional[int] = None,
            replace: bool = False,
            rolling_window: int = None,
            storage: deque = None
    ) -> List[List[BufferedData]]:
        if storage is None:
            storage = self.storage
        if replace:
            sampled_indices = random.choices(range(len(storage)), k=size)
        else:
            try:
                sampled_indices = random.sample(range(len(storage)), k=size)
            except ValueError as e:
                pass
        sampled_data = []
        for idx in sampled_indices:
            slice_ = list(itertools.islice(storage, idx, idx + rolling_window))
            sampled_data.append(slice_)
        return sampled_data

    def _create_index(self, meta_key: str):
        self.meta_index[meta_key] = deque(maxlen=self.storage.maxlen)
        for data in self.storage:
            self.meta_index[meta_key].append(data.meta[meta_key] if meta_key in data.meta else None)

    def __iter__(self) -> deque:
        return iter(self.storage)

    def __copy__(self) -> "DequeBuffer":
        buffer = type(self)(size=self.storage.maxlen)
        buffer.storage = self.storage
        return buffer
