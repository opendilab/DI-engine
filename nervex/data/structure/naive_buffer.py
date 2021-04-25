import copy
import logging
import math
import time
import numbers
from queue import Queue
from typing import Union, NoReturn, Any, Optional, List, Dict
import pickle
import numpy as np
from functools import partial
from easydict import EasyDict

from nervex.data.structure.segment_tree import SumSegmentTree, MinSegmentTree
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from nervex.utils import LockContext, LockContextType, EasyTimer, build_logger


class NaiveReplayBuffer:
    r"""
    Overview:
        Naive replay buffer, can store and sample data.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe, which means that functions like
        ``sample_check``, ``sample``, ``append``, ``extend``, ``clear`` are all mutual to each other.
    Interface:
        __init__, append, extend, sample, update
    Property:
        replay_buffer_size, validlen, beta
    """

    def __init__(
            self,
            name: str,
            replay_buffer_size: int = 10000,
            deepcopy: bool = False,
    ) -> int:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - name (:obj:`str`): Buffer name, mainly used to generate unique data id and logger name.
            - replay_buffer_size (:obj:`int`): The maximum value of the buffer length.
            - deepcopy (:obj:`bool`): Whether to deepcopy data when append/extend and sample data
        """
        self.name = name
        self._replay_buffer_size = replay_buffer_size
        self._deepcopy = deepcopy

        # ``_data`` is a circular queue to store data (or data's reference/file path)
        self._data = [None for _ in range(replay_buffer_size)]
        # Current valid data count, indicating how many elements in ``self._data`` is valid.
        self._valid_count = 0
        # How many pieces of data have been pushed into this buffer, should be no less than ``_valid_count``.
        self._push_count = 0
        # Point to the tail position where next data can be inserted, i.e. latest inserted data's next position.
        self._tail = 0
        # Is used to generate a unique id for each data: If a new data is inserted, its unique id will be this.
        self._next_unique_id = 0
        # Lock to guarantee thread safe
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

    def sample_check(self, size: int, cur_learner_iter: int) -> bool:
        r"""
        Overview:
            Do preparations for sampling and check whther data is enough for sampling
            Preparation includes removing stale transition in ``self._data``.
            Check includes judging whether this buffer satisfies the sample condition:
            current elements count / planning sample count >= min_sample_ratio.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.

        .. note::
            This function must be called exactly before calling ``sample``.
        """
        print(
            '[warning] Naive Buffer does not need to check before sampling, but sample_check method is preserved for compatiblity.'
        )
        return True

    def sample(self, size: int, cur_learner_iter: int) -> Optional[list]:
        r"""
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - sample_data (:obj:`list`): If check fails returns None; Otherwise returns a list with length ``size``, \
                and each data owns keys: original keys + ['IS', 'priority', 'replay_unique_id', 'replay_buffer_idx'].

        .. note::
            Before calling this function, ``sample_check`` must be called.
        """
        if size == 0:
            return []
        with self._lock:
            indices = self._get_indices(size)
            result = self._sample_with_indices(indices, cur_learner_iter)
            return result

    def append(self, ori_data: Any, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Append a data item into queue.
            Add two keys in data:

                - replay_unique_id: The data item's unique id, using ``self._generate_id`` to generate it.
                - replay_buffer_idx: The data item's position index in the queue, this position may already have an \
                    old element, then it would be replaced by this new input one. using ``self._tail`` to locate.
        Arguments:
            - ori_data (:obj:`Any`): The data which will be inserted.
        """
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            self._push_count += 1
            if self._data[self._tail] is None:
                self._valid_count += 1
            data['replay_unique_id'] = self._generate_id(self._next_unique_id)
            data['replay_buffer_idx'] = self._tail
            self._data[self._tail] = data
            self._tail = (self._tail + 1) % self._replay_buffer_size
            self._next_unique_id += 1

    def extend(self, ori_data: List[Any], cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Extend a data list into queue.
            Add two keys in each data item, you can refer to ``append`` for details.
        Arguments:
            - ori_data (:obj:`List[Any]`): The data list.
        """
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            length = len(data)
            # When updating ``_data`` and ``_use_count``, should consider two cases regarding
            # the relationship between "tail + data length" and "replay buffer size" to check whether
            # data will exceed beyond buffer's max length limitation.
            if self._tail + length <= self._replay_buffer_size:
                if self._valid_count != self._replay_buffer_size:
                    self._valid_count += length
                for i in range(length):
                    data[i]['replay_unique_id'] = self._generate_id(self._next_unique_id + i)
                    data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                self._push_count += length
                self._data[self._tail:self._tail + length] = data
            else:
                new_tail = self._tail
                data_start = 0
                residual_num = len(data)
                while True:
                    space = self._replay_buffer_size - new_tail
                    L = min(space, residual_num)
                    if self._valid_count != self._replay_buffer_size:
                        self._valid_count += L
                    for i in range(data_start, data_start + L):
                        data[i]['replay_unique_id'] = self._generate_id(self._next_unique_id + i)
                        data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                    self._push_count += L
                    self._data[new_tail:new_tail + L] = data[data_start:data_start + L]
                    residual_num -= L
                    assert residual_num >= 0
                    if residual_num == 0:
                        break
                    else:
                        new_tail = 0
                        data_start += L
            # Update ``tail`` and ``next_unique_id`` after the whole list is pushed into buffer.
            self._tail = (self._tail + length) % self._replay_buffer_size
            self._next_unique_id += length

    def update(self, info: dict) -> None:
        r"""
        Overview:
            Update a data's priority. Use "repaly_buffer_idx" to locate and "replay_unique_id" to verify.
        Arguments:
            - info (:obj:`dict`): Info dict containing all necessary keys for priority update.
        """
        print('[warning] Naive Buffer does not have priority, therefore calling update method is of no effect.')
        pass

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variables.
        """
        with self._lock:
            for i in range(len(self._data)):
                self._data[i] = None
            self._valid_count = 0
            self._head = 0
            self._max_priority = 1.0

    def close(self) -> None:
        """
        Overview:
            Close the tensorboard logger.
        """
        self.clear()

    def __del__(self) -> None:
        """
        Overview:
            Call ``close`` to delete the object.
        """
        self.close()

    def _get_indices(self, size: int) -> list:
        r"""
        Overview:
            Get the sample index list according to the priority probability.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.
        """
        assert self._valid_count <= self._replay_buffer_size
        if self._valid_count == self._replay_buffer_size:
            tail = self._replay_buffer_size
        else:
            tail = self._tail
        return list(np.random.choice(a=tail, size=size, replace=False))

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int) -> list:
        r"""
        Overview:
            Sample data with ``indices``; Remove a data item if it is used for too many times.
        Arguments:
            - indices (:obj:`List[int]`): A list including all the sample indices.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - data (:obj:`list`) Sampled data.
        """
        data = []
        for idx in indices:
            assert self._data[idx] is not None
            assert self._data[idx]['replay_buffer_idx'] == idx, (self._data[idx]['replay_buffer_idx'], idx)
            if self._deepcopy:
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            data.append(copy_data)
        return data

    def _generate_id(self, data_id: int) -> str:
        """
        Overview:
            Use ``self.name`` and input ``id`` to generate a unique id for next data to be inserted.
        Arguments:
            - data_id (:obj:`int`): Current unique id.
        Returns:
            - id (:obj:`str`): Id in format "BufferName_DataId".
        """
        return "{}_{}".format(self.name, str(data_id))

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

    @property
    def validlen(self) -> int:
        return self._valid_count

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> NoReturn:
        self._beta = beta

    @property
    def used_data(self) -> Any:
        if self._enable_track_used_data:
            if not self._used_data.empty():
                return self._used_data.get()
            else:
                return None
        else:
            return None

    @property
    def push_count(self) -> int:
        return self._push_count

    def state_dict(self) -> dict:
        return {
            'data': self._data,
            'use_count': self._use_count,
            'tail': self._tail,
            'max_priority': self._max_priority,
            'anneal_step': self._anneal_step,
            'beta': self._beta,
            'head': self._head,
            'next_unique_id': self._next_unique_id,
            'valid_count': self._valid_count,
            'sum_tree': self._sum_tree,
            'min_tree': self._min_tree,
        }

    def load_state_dict(self, _state_dict: dict) -> None:
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self.extend(_state_dict['data'])
        else:
            for k, v in _state_dict.items():
                setattr(self, '_{}'.format(k), v)
