import itertools
import random
import uuid
from ditk import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from collections import Counter
from collections import defaultdict, deque, OrderedDict
from ding.data.buffer import Buffer, apply_middleware, BufferedData
from ding.utils import fastcopy
from ding.torch_utils import get_null_data


class BufferIndex():
    """
    Overview:
        Save index string and offset in key value pair.
    """

    def __init__(self, maxlen: int, *args, **kwargs):
        self.maxlen = maxlen
        self.__map = OrderedDict(*args, **kwargs)
        self._last_key = next(reversed(self.__map)) if len(self) > 0 else None
        self._cumlen = len(self.__map)

    def get(self, key: str) -> int:
        value = self.__map[key]
        value = value % self._cumlen + min(0, (self.maxlen - self._cumlen))
        return value

    def __len__(self) -> int:
        return len(self.__map)

    def has(self, key: str) -> bool:
        return key in self.__map

    def append(self, key: str):
        self.__map[key] = self.__map[self._last_key] + 1 if self._last_key else 0
        self._last_key = key
        self._cumlen += 1
        if len(self) > self.maxlen:
            self.__map.popitem(last=False)

    def clear(self):
        self.__map = OrderedDict()
        self._last_key = None
        self._cumlen = 0


class DequeBuffer(Buffer):
    """
    Overview:
        A buffer implementation based on the deque structure.
    """

    def __init__(self, size: int) -> None:
        """
        Overview:
            The initialization method of DequeBuffer.
        Arguments:
            - size (:obj:`int`): The maximum number of objects that the buffer can hold.
        """
        super().__init__(size=size)
        self.storage = deque(maxlen=size)
        self.indices = BufferIndex(maxlen=size)
        # Meta index is a dict which uses deque as values
        self.meta_index = {}

    @apply_middleware("push")
    def push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        """
        Overview:
            The method that input the objects and the related meta information into the buffer.
        Arguments:
            - data (:obj:`Any`): The input object which can be in any format.
            - meta (:obj:`Optional[dict]`): A dict that helps describe data, such as\
                category, label, priority, etc. Default to ``None``.
        """
        return self._push(data, meta)

    @apply_middleware("sample")
    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: Optional[str] = None,
            unroll_len: Optional[int] = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            The method that randomly sample data from the buffer or retrieve certain data by indices.
        Arguments:
            - size (:obj:`Optional[int]`): The number of objects to be obtained from the buffer.
                If ``indices`` is not specified, the ``size`` is required to randomly sample the\
                corresponding number of objects from the buffer.
            - indices (:obj:`Optional[List[str]]`): Only used when you want to retrieve data by indices.
                Default to ``None``.
            - replace (:obj:`bool`): As the sampling process is carried out one by one, this parameter\
                determines whether the previous samples will be put back into the buffer for subsequent\
                sampling. Default to ``False``, it means that duplicate samples will not appear in one\
                ``sample`` call.
            - sample_range (:obj:`Optional[slice]`): The indices range to sample data. Default to ``None``,\
                it means no restrictions on the range of indices for the sampling process.
            - ignore_insufficient (:obj:`bool`): whether throw `` ValueError`` if the sampled size is smaller\
                than the required size. Default to ``False``.
            - groupby (:obj:`Optional[str]`): If this parameter is activated, the method will return a\
                target size of object groups.
            - unroll_len (:obj:`Optional[int]`): The unroll length of a trajectory, used only when the\
                ``groupby`` is activated.
        Returns:
            - sampled_data (Union[List[BufferedData], List[List[BufferedData]]]): The sampling result.
        """
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
        # Groupby and unroll_len
        assert not unroll_len or (
            unroll_len and groupby
        ), "Parameter unroll_len needs to be used in conjunction with groupby."

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
            sampled_data = self._sample_by_group(
                size=size, groupby=groupby, replace=replace, unroll_len=unroll_len, storage=storage
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
                    "Sample operation is ignored due to data insufficient, current buffer is {} while sample is {}".
                    format(self.count(), size)
                )
            else:
                raise ValueError("There are less than {} records/groups in buffer({})".format(size, self.count()))

        sampled_data = self._independence(sampled_data)

        return sampled_data

    @apply_middleware("update")
    def update(self, index: str, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            the method that update data and the related meta information with a certain index.
        Arguments:
            - data (:obj:`Any`): The data which is supposed to replace the old one. If you set it\
                to ``None``, nothing will happen to the old record.
            - meta (:obj:`Optional[dict]`): The new dict which is supposed to merge with the old one.
        """
        if not self.indices.has(index):
            return False
        i = self.indices.get(index)
        item = self.storage[i]
        if data is not None:
            item.data = data
        if meta is not None:
            item.meta = meta
            for key in self.meta_index:
                self.meta_index[key][i] = meta[key] if key in meta else None
        return True

    @apply_middleware("delete")
    def delete(self, indices: Union[str, Iterable[str]]) -> None:
        """
        Overview:
            The method that delete the data and related meta information by specific indices.
        Arguments:
            - indices (Union[str, Iterable[str]]): Where the data to be cleared in the buffer.
        """
        if isinstance(indices, str):
            indices = [indices]
        del_idx = []
        for index in indices:
            if self.indices.has(index):
                del_idx.append(self.indices.get(index))
        if len(del_idx) == 0:
            return
        del_idx = sorted(del_idx, reverse=True)
        for idx in del_idx:
            del self.storage[idx]
        remain_indices = [item.index for item in self.storage]
        key_value_pairs = zip(remain_indices, range(len(indices)))
        self.indices = BufferIndex(self.storage.maxlen, key_value_pairs)

    def count(self) -> int:
        """
        Overview:
            The method that returns the current length of the buffer.
        """
        return len(self.storage)

    def get(self, idx: int) -> BufferedData:
        """
        Overview:
            The method that returns the BufferedData object given a specific index.
        """
        return self.storage[idx]

    @apply_middleware("clear")
    def clear(self) -> None:
        """
        Overview:
            The method that clear all data, indices, and the meta information in the buffer.
        """
        self.storage.clear()
        self.indices.clear()
        self.meta_index = {}

    def import_data(self, data_with_meta: List[Tuple[Any, dict]]) -> None:
        """
        Overview:
            The method that push data by sequence.
        Arguments:
            data_with_meta (List[Tuple[Any, dict]]): The sequence of (data, meta) tuples.
        """
        for data, meta in data_with_meta:
            self._push(data, meta)

    def export_data(self) -> List[BufferedData]:
        """
        Overview:
            The method that export all data in the buffer by sequence.
        Returns:
            storage (List[BufferedData]): All ``BufferedData`` objects stored in the buffer.
        """
        return list(self.storage)

    def _push(self, data: Any, meta: Optional[dict] = None) -> BufferedData:
        index = uuid.uuid1().hex
        if meta is None:
            meta = {}
        buffered = BufferedData(data=data, index=index, meta=meta)
        self.storage.append(buffered)
        self.indices.append(index)
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
                can be nested if groupby has been set.
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

    def _sample_by_group(
            self,
            size: int,
            groupby: str,
            replace: bool = False,
            unroll_len: Optional[int] = None,
            storage: deque = None
    ) -> List[List[BufferedData]]:
        """
        Overview:
            Sampling by `group` instead of records, the result will be a collection
            of lists with a length of `size`, but the length of each list may be different from other lists.
        """
        if storage is None:
            storage = self.storage
        if groupby not in self.meta_index:
            self._create_index(groupby)

        def filter_by_unroll_len():
            "Filter groups by unroll len, ensure count of items in each group is greater than unroll_len."
            group_count = Counter(self.meta_index[groupby])
            group_names = []
            for key, count in group_count.items():
                if count >= unroll_len:
                    group_names.append(key)
            return group_names

        if unroll_len and unroll_len > 1:
            group_names = filter_by_unroll_len()
        else:
            group_names = list(set(self.meta_index[groupby]))

        sampled_groups = []
        if replace:
            sampled_groups = random.choices(group_names, k=size)
        else:
            try:
                sampled_groups = random.sample(group_names, k=size)
            except ValueError:
                raise ValueError("There are less than {} groups in buffer({} groups)".format(size, len(group_names)))

        # Build dict like {"group name": [records]}
        sampled_data = defaultdict(list)
        for buffered in storage:
            meta_value = buffered.meta[groupby] if groupby in buffered.meta else None
            if meta_value in sampled_groups:
                sampled_data[buffered.meta[groupby]].append(buffered)

        final_sampled_data = []
        for group in sampled_groups:
            seq_data = sampled_data[group]
            # Filter records by unroll_len
            if unroll_len:
                start_indice = random.choice(range(max(1, len(seq_data) - unroll_len)))
                seq_data = seq_data[start_indice:start_indice + unroll_len]
            final_sampled_data.append(seq_data)

        return final_sampled_data

    def _create_index(self, meta_key: str):
        self.meta_index[meta_key] = deque(maxlen=self.storage.maxlen)
        for data in self.storage:
            self.meta_index[meta_key].append(data.meta[meta_key] if meta_key in data.meta else None)

    def __iter__(self) -> deque:
        return iter(self.storage)

    def __copy__(self) -> "DequeBuffer":
        buffer = type(self)(size=self.storage.maxlen)
        buffer.storage = self.storage
        buffer.meta_index = self.meta_index
        buffer.indices = self.indices
        return buffer
