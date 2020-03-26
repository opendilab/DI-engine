import numpy as np


class BaseBuffer(object):
    def __init__(self, maxlen, reuse_max_limit=None):
        self.maxlen = maxlen
        self._data = [None for _ in range(maxlen)]
        self._reuse_count = [0 for _ in range(maxlen)]
        self._valid = []
        self.reuse_max_limit = reuse_max_limit if reuse_max_limit is not None else np.inf
        self.pointer = 0

    def append(self, data):
        self._data[self.pointer] = data
        self._reuse_count[self.pointer] = 0
        self._valid.append(self.pointer)
        self.pointer = (self.pointer + 1) % self.maxlen

    def extend(self, data):
        L = len(data)
        if self.pointer + L <= self.maxlen:
            self._data[self.pointer:self.pointer + L] = data
            self._reuse_count[self.pointer:self.pointer + L] = [0 for _ in range(L)]
            self._valid.extend([i for i in range(self.pointer, self.pointer + L)])
        else:
            mid = self.maxlen - self.pointer
            self._data[self.pointer:self.pointer + mid] = data[:mid]
            self._data[:L - mid] = data[mid:]
            self._reuse_count[self.pointer:self.pointer + mid] = [0 for _ in range(mid)]
            self._reuse_count[:L - mid] = [0 for _ in range(L - mid)]
            self._valid.extend([(i % self.maxlen) for i in range(self.pointer, self.pointer + L)])

        self.pointer = (self.pointer + L) % self.maxlen

    def sample(self, size):
        if size > len(self._valid):
            raise Exception("no enough element for sample(expect: {}/current have: {})".format(size, len(self._valid)))

        valid_indices = np.random.choice(self._valid, size, replace=False)
        data_indices = []
        for idx in valid_indices:
            data_idx = self._valid[idx]
            data_indices.append(self._data[data_idx])
            self._reuse_count[data_idx] += 1
            # remove the item which reuse is bigger than reuse_max_limit
            if self._reuse_count[data_idx] > self.reuse_max_limit:
                self._data[data_idx] = None
                del self._valid[idx]
        return data_indices
