import copy
from typing import Union, Any, Optional, List
import numpy as np

from nervex.worker.replay_buffer import BaseBuffer
from nervex.utils import LockContext, LockContextType, BUFFER_REGISTRY
from .utils import UsedDataRemover, generate_id


@BUFFER_REGISTRY.register('naive')
class NaiveReplayBuffer(BaseBuffer):
    r"""
    Overview:
        Naive replay buffer, can store and sample data.
        An naive implementation of replay buffer with no priority and any other advanced features.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe, which means that functions like
        ``sample_check``, ``sample``, ``append``, ``extend``, ``clear`` are all mutual to each other.
    Interface:
        __init__, start, close, push, update, sample, clear, count, state_dict, load_state_dict
    Property:
        replay_buffer_size, validlen, beta
    """

    config = dict(
        type='naive',
        name='default_buffer',
        replay_buffer_size=10000,
        deepcopy=False,
        # default `False` for serial pipeline
        enable_track_used_data=False,
    )

    def __init__(
            self,
            name: str,
            cfg: dict,
    ) -> None:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - name (:obj:`str`): Buffer name, used to generate unique data id and logger name.
            - cfg (:obj:`dict`): cfg dict
            - deepcopy (:obj:`bool`): Whether to deepcopy data when append/extend and sample data
        """
        self.name = name
        self._cfg = cfg
        self._replay_buffer_size = self._cfg.replay_buffer_size
        self._deepcopy = self._cfg.deepcopy
        # ``_data`` is a circular queue to store data (full data or meta data)
        self._data = [None for _ in range(self._replay_buffer_size)]
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
        self._end_flag = False
        self._enable_track_used_data = self._cfg.enable_track_used_data
        if self._enable_track_used_data:
            self._used_data_remover = UsedDataRemover()

    def start(self) -> None:
        if self._enable_track_used_data:
            self._used_data_remover.start()
        else:
            print('[BUFFER WARNING] `enable_track_used_data` is False, no thread to start.`')

    def close(self) -> None:
        if self._enable_track_used_data:
            self._used_data_remover.close()
        else:
            print('[BUFFER WARNING] `enable_track_used_data` is False, no thread to join.`')

    def push(self, data: Union[list, dict], cur_collector_envstep: int) -> None:
        if isinstance(data, list):
            self._extend(data, cur_collector_envstep)
        else:
            self._append(data, cur_collector_envstep)

    def sample(self, size: int, cur_learner_iter: int) -> Optional[list]:
        r"""
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Not used in this method, but preserved for compatiblity.
        Returns:
            - sample_data (:obj:`list`): A list of data with length ``size``.
        """
        if size == 0:
            return []
        with self._lock:
            indices = self._get_indices(size)
            result = self._sample_with_indices(indices, cur_learner_iter)
            return result

    def _append(self, ori_data: Any, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Append a data item into ``self._data``.
            Add two keys in data:

                - replay_unique_id: The data item's unique id, using ``generate_id`` to generate it.
                - replay_buffer_idx: The data item's position index in the queue, this position may already have an \
                    old element, then it would be replaced by this new input one. using ``self._tail`` to locate.
        Arguments:
            - ori_data (:obj:`Any`): The data which will be inserted.
            - cur_collector_envstep (:obj:`int`): Not used in this method, but preserved for compatiblity.
        """
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            self._push_count += 1
            if self._data[self._tail] is None:
                self._valid_count += 1
            elif self._enable_track_used_data:
                self._used_data_remover.add_used_data(self._data[self._tail])
            data['replay_unique_id'] = generate_id(self.name, self._next_unique_id)
            data['replay_buffer_idx'] = self._tail
            self._data[self._tail] = data
            self._tail = (self._tail + 1) % self._replay_buffer_size
            self._next_unique_id += 1

    def _extend(self, ori_data: List[Any], cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Extend a data list into queue.
            Add two keys in each data item, you can refer to ``_append`` for details.
        Arguments:
            - ori_data (:obj:`List[Any]`): The data list.
            - cur_collector_envstep (:obj:`int`): Not used in this method, but preserved for compatiblity.
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
                elif self._enable_track_used_data:
                    for i in range(length):
                        self._used_data_remover.add_used_data(self._data[self._tail + i])
                for i in range(length):
                    data[i]['replay_unique_id'] = generate_id(self.name, self._next_unique_id + i)
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
                    elif self._enable_track_used_data:
                        for i in range(L):
                            self._used_data_remover.add_used_data(self._data[new_tail + i])
                    for i in range(data_start, data_start + L):
                        data[i]['replay_unique_id'] = generate_id(self.name, self._next_unique_id + i)
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
            Naive Buffer does not need to update any info, but this method is preserved for compatiblity.
        """
        print(
            '[BUFFER WARNING] Naive Buffer does not need to update any info, \
                but `update` method is preserved for compatiblity.'
        )

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variables.
        """
        with self._lock:
            for i in range(len(self._data)):
                if self._data[i] is not None:
                    self._used_data_remover.add_used_data(self._data[i])
                    self._data[i] = None
            self._valid_count = 0

    def __del__(self) -> None:
        """
        Overview:
            Call ``close`` to delete the object.
        """
        self.close()

    def _get_indices(self, size: int) -> list:
        r"""
        Overview:
            Get the sample index list.
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
            Sample data with ``indices``.
        Arguments:
            - indices (:obj:`List[int]`): A list including all the sample indices.
            - cur_learner_iter (:obj:`int`): Not used in this method, but preserved for compatiblity.
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

    def count(self) -> int:
        return self._valid_count

    def state_dict(self) -> dict:
        return {
            'data': self._data,
            'tail': self._tail,
            'next_unique_id': self._next_unique_id,
            'valid_count': self._valid_count,
            'push_count': self._push_count,
        }

    def load_state_dict(self, _state_dict: dict) -> None:
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self._extend(_state_dict['data'])
        else:
            for k, v in _state_dict.items():
                setattr(self, '_{}'.format(k), v)

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

    @property
    def validlen(self) -> int:
        return self._valid_count

    @property
    def push_count(self) -> int:
        return self._push_count
