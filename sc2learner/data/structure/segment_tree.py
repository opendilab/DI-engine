import numpy as np


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element=None):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.operation = operation
        if neutral_element is None:
            if operation is sum:
                neutral_element = 0.
            elif operation is min:
                neutral_element = np.inf
            elif operation is max:
                neutral_element = -np.inf
        self.neutral_element = neutral_element
        self.value = [self.neutral_element for _ in range(2 * capacity)]

    def reduce(self, start=0, end=None):
        '''
        Note:
            [start, end)
        '''
        if end is None:
            end = self.capacity
        assert (start < end)

        start += self.capacity
        end += self.capacity
        result = self.neutral_element

        while start < end:
            if start & 1:
                result = self.operation([result, self.value[start]])
                start += 1
            if end & 1:
                end -= 1
                result = self.operation([result, self.value[end]])

            start = start >> 1
            end = end >> 1
        return result

    def __setitem__(self, idx, val):
        assert (0 <= idx <= self.capacity)
        idx += self.capacity
        self.value[idx] = val

        idx = idx >> 1  # transform to father node idx
        while idx >= 1:
            child_base = 2 * idx
            self.value[idx] = self.operation([self.value[child_base], self.value[child_base + 1]])
            idx = idx >> 1

    def __getitem__(self, idx):
        assert (0 <= idx <= self.capacity)
        return self.value[idx + self.capacity]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity, operation=sum)

    def find_prefixsum_idx(self, prefixsum, trust_caller=True):
        '''
        Overview: find the highest index i, which for j in 0 <= j <= i, \sum_{j}leaf[j] <= prefixsum
        '''
        if not trust_caller:
            assert 0 <= prefixsum <= self.reduce() + 1e-5
        idx = 1  # parent node
        while idx < self.capacity:  # non-leaf node
            child_base = 2 * idx
            if self.value[child_base] > prefixsum:
                idx = child_base
            else:
                prefixsum -= self.value[child_base]
                idx = child_base + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity, operation=min)
