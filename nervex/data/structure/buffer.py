import copy
import math
import time
import numbers
from queue import Queue
from typing import Union, NoReturn, Any, Optional, List

import numpy as np

from nervex.data.structure.segment_tree import SumSegmentTree, MinSegmentTree


class RecordList(list):
    r"""
    Overview:
        A list which can track old elements when being set to a new one.
    Interface:
        __init__, __setitem__
    """

    def __init__(self, *args, **kwargs) -> None:
        r"""
        Overview:
            Additionally init a queue, which is used to track old replaced elements.
        """
        super(RecordList, self).__init__(*args, **kwargs)
        self._used_data = Queue()

    def __setitem__(self, idx: Union[numbers.Integral, slice], data: Any) -> None:
        r"""
        Overview:
            Additionally append the replaced element(if not None) to ``self._used_data``
            to track it for further operation.
        Arguments:
            - idx (:obj:`Union[numbers.Integral, slice]`): one or many, indicating the index where \
                    the element will be replaced by a new one
            - data (:obj:`Any`): one/many(depending on ``idx``) pieces of data to be inserted to the list
        """
        if isinstance(idx, numbers.Integral):
            if self[idx] is not None:
                self._used_data.put(self[idx])
        elif isinstance(idx, slice):
            old_data = self[idx]
            for d in old_data:
                if d is not None:
                    self._used_data.put(d)
        else:
            raise TypeError("not support idx type: {}".format(type(idx)))
        super().__setitem__(idx, data)


class PrioritizedBuffer:
    r"""
    Overview:
        Prioritized buffer, can store and sample data
    Interface:
        __init__, append, extend, sample, update
    Property:
        maxlen, validlen, beta
    Note:
        This buffer doesn't refer to multi-thread/multi-process, thread-safe should be ensured by the caller
    """

    def __init__(
        self,
        maxlen: int,
        max_reuse: Union[int, None] = None,
        max_staleness: Union[int, None] = None,
        min_sample_ratio: float = 1.,
        alpha: float = 0.,
        beta: float = 0.,
        anneal_step: int = 0,
        eps: float = 0.01,
        enable_track_used_data: bool = False,
        deepcopy: bool = False
    ):
        r"""
        Overview:
            Initialize the buffer
        Arguments:
            - maxlen (:obj:`int`): the maximum value of the buffer length
            - max_reuse (:obj:`int` or None): the maximum reuse times of each element in buffer
            - min_sample_ratio (:obj:`float`): the minimum ratio restriction for sampling, only when \
                "current element number in buffer divided by sample size" is greater than this, can start sampling
            - alpha (:obj:`float`): how much prioritization is used (0: no prioritization, 1: full prioritization)
            - beta (:obj:`float`): how much correction is used (0: no correction, 1: full correction)
            - anneal_step (:obj:`int`): anneal step for beta(beta -> 1)
            - eps (:obj:`float`): small positive number for avoiding edge-case
            - enable_track_used_data (:obj:`bool`): whether tracking the used data
            - deepcopy (:obj:`bool`): whether deepcopy data when append/extend and sample data
        """
        # TODO(nyz) remove elements according to priority
        # TODO(nyz) add statistics module
        self._maxlen = maxlen
        self._enable_track_used_data = enable_track_used_data
        self._deepcopy = deepcopy
        # use RecordList if needs to track used data; otherwise use normal list
        if self._enable_track_used_data:
            self._data = RecordList([None for _ in range(maxlen)])
        else:
            self._data = [None for _ in range(maxlen)]
        self._reuse_count = {idx: 0 for idx in range(maxlen)}  # {position_idx: reuse_count}

        self.max_reuse = max_reuse if max_reuse is not None else np.inf
        self.max_staleness = max_staleness if max_staleness is not None else np.inf
        assert (min_sample_ratio >= 1)
        self.min_sample_ratio = min_sample_ratio
        assert (0 <= alpha <= 1)
        self.alpha = alpha
        assert (0 <= beta <= 1)
        self._beta = beta
        self._anneal_step = anneal_step
        if self._anneal_step != 0:
            self._beta_anneal_step = (1 - self._beta) / self._anneal_step
        self._eps = eps
        # capacity needs to be the power of 2
        capacity = int(np.power(2, np.ceil(np.log2(self.maxlen))))
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        self.max_priority = 1.0
        # current valid data count
        self._valid_count = 0
        # todo
        self._push_count = 0
        # pointing to the position where last data item is
        self.pointer = 0
        # generate a unique id for each data
        self.latest_data_id = 0

        # data check function list, used in ``append`` and ``extend``
        self.check_list = [lambda x: isinstance(x, dict)]

    def _set_weight(self, idx: int, data: Any) -> None:
        r"""
        Overview:
            Set the priority and tree weight of the input data
        Arguments:
            - idx (:obj:`int`): the index of the list where the data will be inserted
            - data (:obj:`Any`): the data which will be inserted
        """
        if 'priority' not in data.keys() or data['priority'] is None:
            data['priority'] = self.max_priority
        weight = data['priority'] ** self.alpha
        self.sum_tree[idx] = weight
        self.min_tree[idx] = weight

    def sample(self, size: int, cur_learner_iter: int) -> Optional[list]:
        r"""
        Overview:
            Sample data with length ``size``
        Arguments:
            - size (:obj:`int`): the number of the data that will be sampled
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness
        Returns:
            - sample_data (:obj:`list`): If check fails returns None; Otherwise returns a list with length ``size``, \
                and each data owns keys: original keys + ['IS', 'priority', 'replay_unique_id', 'replay_buffer_idx']
        """
        # todo: while loop here, to do from here
        left_to_sample = size
        result = []
        while left_to_sample:
            if not self._sample_check(left_to_sample):
                return None
            indices = self._get_indices(left_to_sample)
            result += self._sample_with_indices(indices, cur_learner_iter, len(result) == 0)
            left_to_sample = size - len(result)
        # Deepcopy ``result``'s same indice data in case ``self._get_indices`` may get datas with the same indices.
        for i in range(size):
            tmp = []
            for j in range(i + 1, size):
                if id(result[i]) == id(result[j]):
                    tmp.append(j)
            for j in tmp:
                result[j] = copy.deepcopy(result[j])
        return result

    def append(self, ori_data: Any) -> None:
        r"""
        Overview:
            Append a data item into queue. Add two keys in data:

                - replay_unique_id: the data item's unique id, using ``self.latest_data_id`` to generate it
                - replay_buffer_idx: the data item's position index in the queue, this position may had an \
                    old element but wass replaced by this input new one. using ``self.pointer`` to generate it
        Arguments:
            - ori_data (:obj:`Any`): the data which will be inserted
        """
        if self._deepcopy:
            data = copy.deepcopy(ori_data)
        else:
            data = ori_data
        try:
            assert (self._data_check(data))
        except AssertionError:
            # if data check fails, just return without any operations
            print('illegal data {}, reject it...'.format(type(data)))
            return
        if self._data[self.pointer] is None:
            self._valid_count += 1
        self._push_count += 1
        data['replay_unique_id'] = self.latest_data_id
        data['replay_buffer_idx'] = self.pointer
        self._set_weight(self.pointer, data)
        self._data[self.pointer] = data
        self._reuse_count[self.pointer] = 0
        self.pointer = (self.pointer + 1) % self._maxlen
        self.latest_data_id += 1

    def extend(self, ori_data: List[Any]) -> None:
        r"""
        Overview:
            Extend a data list into queue. Add two keys in each data item, you can reference ``append`` for details.
        Arguments:
            - ori_data (:obj:`T`): the data list
        """
        if self._deepcopy:
            data = copy.deepcopy(ori_data)
        else:
            data = ori_data
        check_result = [self._data_check(d) for d in data]
        # only keep data items that pass data_check
        valid_data = [d for d, flag in zip(data, check_result) if flag]
        length = len(valid_data)
        for i in range(length):
            valid_data[i]['replay_unique_id'] = self.latest_data_id + i
            valid_data[i]['replay_buffer_idx'] = (self.pointer + i) % self.maxlen
            self._set_weight((self.pointer + i) % self.maxlen, valid_data[i])
            if self._data[(self.pointer + i) % self.maxlen] is None:
                self._valid_count += 1
            self._push_count += 1
        # when updating ``_data`` and ``_reuse_count``, should consider
        # two cases of the relationship between "pointer + data length" and "queue max length" to check whether
        # data will exceed beyond queue's max length limitation
        if self.pointer + length <= self._maxlen:
            self._data[self.pointer:self.pointer + length] = valid_data
            for idx in range(self.pointer, self.pointer + length):
                self._reuse_count[idx] = 0
        else:
            data_start = self.pointer
            valid_data_start = 0
            residual_num = len(valid_data)
            while True:
                space = self._maxlen - data_start
                L = min(space, residual_num)
                self._data[data_start:data_start + L] = valid_data[valid_data_start:valid_data_start + L]
                residual_num -= L
                for i in range(data_start, data_start + L):
                    self._reuse_count[i] = 0
                if residual_num <= 0:
                    break
                else:
                    data_start = 0

        # update ``pointer`` and ``latest_data_id`` after the whole list is inserted
        self.pointer = (self.pointer + length) % self._maxlen
        self.latest_data_id += length

    def update(self, info: dict):
        r"""
        Overview:
            Update priority according to the id and idx
        Arguments:
            - info (:obj:`dict`): info dict containing all necessary keys for priority update
        """
        data = [info['replay_unique_id'], info['replay_buffer_idx'], info['priority']]
        for id_, idx, priority in zip(*data):
            # if the data still exists in the queue, then do the update operation
            if self._data[idx] is not None \
                    and self._data[idx]['replay_unique_id'] == id_:  # confirm the same transition(data)
                assert priority >= 0, priority
                self._data[idx]['priority'] = priority + self._eps
                self._set_weight(idx, self._data[idx])
                # update max priority
                self.max_priority = max(self.max_priority, priority)

    def _data_check(self, d) -> bool:
        r"""
        Overview:
            Data legality check, using rules in ``self.check_list``
        Arguments:
            - d (:obj:`T`): the data which needs to be checked
        Returns:
            - result (:obj:`bool`): whether the data passes the check
        """
        # only the data passes all the check functions, would the check return True
        return all([fn(d) for fn in self.check_list])

    def _sample_check(self, size: int) -> bool:
        r"""
        Overview:
            Check whether the buffer satisfies the sample condition (current elements are enough for sampling)
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - result (:obj:`bool`): Whether the buffer can sample
        """
        if self._valid_count / size < self.min_sample_ratio:
            print(
                "[INFO({})] no enough element for sample(expect: {}/current have: {}, min_sample_ratio: {})".format(
                    time.time(), size, self._valid_count, self.min_sample_ratio
                )
            )
            return False
        else:
            return True

    def _get_indices(self, size: int) -> list:
        r"""
        Overview:
            Get the sample index list according to the priority probability,
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - index_list (:obj:`list`): A list including all the sample indices
        """
        # average divide size intervals and sample from them
        intervals = np.array([i * 1.0 / size for i in range(size)])
        # uniform sample in each interval
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        # rescale to [0, S), where S is the sum of the total sum_tree
        mass *= self.sum_tree.reduce()
        # find prefix sum index to approximate sample with probability
        return [self.sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int) -> None:
        self._data[idx] = None
        self.sum_tree[idx] = self.sum_tree.neutral_element
        self.min_tree[idx] = self.min_tree.neutral_element
        self._valid_count -= 1

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int, first_sample: bool = True) -> list:
        r"""
        Overview:
            Sample data with ``indices``; If a data item is reused for too many times,
            remove it and update internal variables(sum_tree, min_tree, valid_count)
        Arguments:
            - indices (:obj:`List[int]`): A list including all the sample indices
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness
        Returns:
            - data (:obj:`list`) Sampled data
        """
        # calculate max weight for normalizing IS
        sum_tree_root = self.sum_tree.reduce()
        p_min = self.min_tree.reduce() / sum_tree_root
        max_weight = (self._valid_count * p_min) ** (-self._beta)
        data = []
        for idx in indices:
            # calculate staleness, if too stale, remove it and do not add it to the return data
            staleness = cur_learner_iter - self._data[idx].get('collect_iter', cur_learner_iter)
            if staleness >= self.max_staleness:
                self._remove(idx)
                continue
            if self._deepcopy:
                # deepcopy data for avoiding interference
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            assert (copy_data is not None)
            copy_data['staleness'] = staleness
            # store reuse for outer monitor
            copy_data['reuse'] = self._reuse_count[idx]
            # get IS(importance sampling weight for gradient step)
            p_sample = self.sum_tree[copy_data['replay_buffer_idx']] / sum_tree_root
            weight = (self._valid_count * p_sample) ** (-self._beta)
            copy_data['IS'] = weight / max_weight
            data.append(copy_data)
            self._reuse_count[idx] += 1
        # remove the item whose "reuse count" is greater than max_reuse
        for idx in indices:
            if self._reuse_count[idx] > self.max_reuse:
                self._remove(idx)
        # anneal update beta, only the first sample will update, later samples caused by staleness will not update
        # because they are together belong to one actual sample
        if first_sample and self._anneal_step != 0:
            self._beta += self._beta_anneal_step
        return data

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variable
        """
        for i in range(len(self._data)):
            self._remove(i)
            self._reuse_count[i] = 0
        self._valid_count = 0
        self.pointer = 0
        self.max_priority = 1.0

    @property
    def maxlen(self) -> int:
        return self._maxlen

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
            if not self._data._used_data.empty():
                return self._data._used_data.get()
            else:
                return None
        else:
            return None

    @property
    def push_count(self) -> int:
        return self._push_count
