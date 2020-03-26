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
            - beta (:obj:`float`):
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

    def _get_IS(self, data):
        '''
        Note: get the importance sampling weight for gradient step
        '''
        if self.use_priority:
            sum_tree_root = self.sum_tree.reduce()
            p_min = self.min_tree.reduce() / sum_tree_root
            max_weight = (self.valid_count * p_min)**(-self.beta)
            for d in data:
                p_sample = self.sum_tree[d['replay_buffer_idx']] / sum_tree_root
                weight = (self.valid_count * p_sample)**(-self.beta)
                d['IS'] = weight / max_weight
        else:
            for d in data:
                d['IS'] = 1.0
        return data

    def sample(self, size):
        '''
        Returns:
            - sample_data (:obj:`list`): each data owns keys:
                original data keys + ['IS', 'priority', 'replay_buffer_id', 'replay_buffer_idx]'
        '''
        self._sample_check(size)
        indices = self._get_indices(size)
        sample_data = self._sample_with_indices(indices)
        sample_data = self._get_IS(sample_data)
        return sample_data

    def append(self, data):
        assert (self._data_check(data))
        data['replay_buffer_id'] = self.data_id
        data['replay_buffer_idx'] = self.pointer
        self._set_weight(self.pointer, data)
        self._data[self.pointer] = data
        self._reuse_count[self.pointer] = 0
        self.pointer = (self.pointer + 1) % self._maxlen
        self.valid_count += 1
        self.data_id += 1

    def extend(self, data):
        assert (all([self._data_check(d) for d in data]))
        L = len(data)
        for i in range(L):
            data[i]['replay_buffer_id'] = self.data_id + i
            data[i]['replay_buffer_idx'] = (self.pointer + i) % self.maxlen
            self._set_weight((self.pointer + i) % self.maxlen, data[i])

        if self.pointer + L <= self._maxlen:
            self._data[self.pointer:self.pointer + L] = data
            self._reuse_count[self.pointer:self.pointer + L] = [0 for _ in range(L)]
            self._valid.extend([i for i in range(self.pointer, self.pointer + L)])
        else:
            mid = self._maxlen - self.pointer
            self._data[self.pointer:self.pointer + mid] = data[:mid]
            self._data[:L - mid] = data[mid:]
            self._reuse_count[self.pointer:self.pointer + mid] = [0 for _ in range(mid)]
            self._reuse_count[:L - mid] = [0 for _ in range(L - mid)]
            self._valid.extend([(i % self._maxlen) for i in range(self.pointer, self.pointer + L)])

        self.pointer = (self.pointer + L) % self._maxlen
        self.valid_count += L
        self.data_id += L

    def update(self, info):
        for id, idx, priority in zip(*info.values()):
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
            raise Exception(
                "no enough element for sample(expect: {}/current have: {}, min_sample_ratio: {})".format(
                    size, self.valid_count, self.min_sample_ratio
                )
            )

    def _sample_with_indices(self, indices):
        data = []
        for idx in indices:
            data.append(copy.deepcopy(self._data[idx]))
            self._reuse_count[idx] += 1
            # remove the item which reuse is bigger than max_reuse
            if self._reuse_count[idx] > self.max_reuse:
                self._data[idx] = None
                self.sum_tree[idx] = 0.
                self.min_tree[idx] = 0.
                self.valid_count -= 1
        return data

    @property
    def maxlen(self):
        return self._maxlen

    @property
    def validlen(self):
        return self.valid_count
