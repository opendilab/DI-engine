import numpy as np
import copy
from .segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedBuffer(object):
    '''
    Interfacce: __init__, append, extend, sample, update
    '''
    def __init__(self, maxlen, max_reuse=None, min_sample_ratio=1., alpha=0., beta=0.):
        '''
        Arguments:
            - maxlen (:obj:`int`): the maximum value of the buffer length
            - max_reuse (:obj:`int` or None): the maximum reuse times of each element in buffer
            - min_sample_ratio (:obj:`float`) : the minimum ratio of the current element size in buffer
                                                divides sample size
            - alpha (:obj:`float`): how much prioritization is used(0: no prioritization, 1: full prioritization)
            - beta (:obj:`float`): how much correction is used(0: no correction, 1: full correction)
        '''
        # TODO(nyz) remove elements according to priority
        # TODO(nyz) whether use Lock
        self._maxlen = maxlen
        self._data = [None for _ in range(maxlen)]
        self._reuse_count = [0 for _ in range(maxlen)]

        self.max_reuse = max_reuse if max_reuse is not None else np.inf
        assert (min_sample_ratio >= 1)
        self.min_sample_ratio = min_sample_ratio
        assert (0 <= alpha <= 1)
        self.alpha = alpha
        assert (0 <= beta <= 1)
        self._beta = beta
        capacity = int(np.power(2, np.ceil(np.log2(self.maxlen))))
        self.sum_tree = SumSegmentTree(capacity)
        self.use_priority = np.fabs(alpha) > 1e-4
        if self.use_priority:  # for simplifying runtime execution of the no priority buffer
            self.min_tree = MinSegmentTree(capacity)

        self.max_priority = 1.0
        self.valid_count = 0
        self.pointer = 0
        self.data_id = 0

        self.check_list = [lambda x: isinstance(x, dict)]

    def _set_weight(self, idx, data):
        if 'priority' not in data.keys() or data['priority'] is None:
            data['priority'] = self.max_priority
        weight = data['priority']**self.alpha
        self.sum_tree[idx] = weight
        if self.use_priority:
            self.min_tree[idx] = weight

    def sample(self, size):
        '''
        Returns:
            - sample_data (:obj:`list`): each data owns keys:
                original data keys + ['IS', 'priority', 'replay_buffer_id', 'replay_buffer_idx]'
        '''
        if not self._sample_check(size):
            return None
        indices = self._get_indices(size)
        return self._sample_with_indices(indices)

    def append(self, data):
        try:
            assert (self._data_check(data))
        except AssertionError:
            return
        if self._data[self.pointer] is None:
            self.valid_count += 1
        data['replay_buffer_id'] = self.data_id
        data['replay_buffer_idx'] = self.pointer
        self._set_weight(self.pointer, data)
        self._data[self.pointer] = data
        self._reuse_count[self.pointer] = 0
        self.pointer = (self.pointer + 1) % self._maxlen
        self.data_id += 1

    def extend(self, data):
        check_result = [self._data_check(d) for d in data]
        valid_data = [d for d, flag in zip(data, check_result) if flag]
        L = len(valid_data)
        for i in range(L):
            valid_data[i]['replay_buffer_id'] = self.data_id + i
            valid_data[i]['replay_buffer_idx'] = (self.pointer + i) % self.maxlen
            self._set_weight((self.pointer + i) % self.maxlen, valid_data[i])
            if self._data[(self.pointer + i) % self.maxlen] is None:
                self.valid_count += 1

        if self.pointer + L <= self._maxlen:
            self._data[self.pointer:self.pointer + L] = valid_data
            self._reuse_count[self.pointer:self.pointer + L] = [0 for _ in range(L)]
        else:
            mid = self._maxlen - self.pointer
            self._data[self.pointer:self.pointer + mid] = valid_data[:mid]
            self._data[:L - mid] = valid_data[mid:]
            self._reuse_count[self.pointer:self.pointer + mid] = [0 for _ in range(mid)]
            self._reuse_count[:L - mid] = [0 for _ in range(L - mid)]

        self.pointer = (self.pointer + L) % self._maxlen
        self.data_id += L

    def update(self, info):
        data = [info['replay_buffer_id'], info['replay_buffer_idx'], info['priority']]
        for id, idx, priority in zip(*data):
            if self._data[idx]['replay_buffer_id'] == id:  # confirm the same transition
                assert priority > 0
                self._data[idx]['priority'] = priority
                self._set_weight(idx, self._data[idx])
                self.max_priority = max(self.max_priority, priority)

    def _data_check(self, d):
        return all([fn(d) for fn in self.check_list])

    def _get_indices(self, size):
        # average divide size intervals and sample from them
        intervals = np.array([i * 1.0 / size for i in range(size)])
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        mass *= self.sum_tree.reduce()
        return [self.sum_tree.find_prefixsum_idx(m) for m in mass]

    def _sample_check(self, size):
        if self.valid_count / size < self.min_sample_ratio:
            print(
                "no enough element for sample(expect: {}/current have: {}, min_sample_ratio: {})".format(
                    size, self.valid_count, self.min_sample_ratio
                )
            )
            return False
        else:
            return True

    def _sample_with_indices(self, indices):
        if self.use_priority:
            sum_tree_root = self.sum_tree.reduce()
            p_min = self.min_tree.reduce() / sum_tree_root
            max_weight = (self.valid_count * p_min)**(-self._beta)

        data = []
        for idx in indices:
            copy_data = copy.deepcopy(self._data[idx])
            assert (copy_data is not None)
            # get IS(importance sampling weight for gradient step)
            if self.use_priority:
                p_sample = self.sum_tree[copy_data['replay_buffer_idx']] / sum_tree_root
                weight = (self.valid_count * p_sample)**(-self._beta)
                copy_data['IS'] = weight / max_weight
            else:
                copy_data['IS'] = 1.0
            data.append(copy_data)
            self._reuse_count[idx] += 1
        # remove the item which reuse is bigger than max_reuse
        for idx in indices:
            if self._reuse_count[idx] > self.max_reuse:
                self._data[idx] = None
                self.sum_tree[idx] = self.sum_tree.neutral_element
                if self.use_priority:
                    self.min_tree[idx] = self.min_tree.neutral_element
                self.valid_count -= 1
        return data

    @property
    def maxlen(self):
        return self._maxlen

    @property
    def validlen(self):
        return self.valid_count

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
