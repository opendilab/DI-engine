import numpy as np


class SegmentTree:
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
            else:
                raise ValueError("operation argument should be in min, max, sum (built in python functions).")
        self.neutral_element = neutral_element
        self.value = [self.neutral_element for _ in range(2 * capacity)]

    def reduce(self, start=0, end=None):
        """
        Note:
            [start, end)
        """
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
        assert (0 <= idx < self.capacity)
        idx += self.capacity
        self.value[idx] = val

        idx = idx >> 1  # transform to father node idx
        while idx >= 1:
            child_base = 2 * idx
            self.value[idx] = self.operation([self.value[child_base], self.value[child_base + 1]])
            idx = idx >> 1

    def __getitem__(self, idx):
        assert (0 <= idx < self.capacity)
        return self.value[idx + self.capacity]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity, operation=sum)

    def find_prefixsum_idx(self, prefixsum, trust_caller=True):
        """
        Overview: find the highest non-zero index i, which for j in 0 <= j < i, sum_{j}leaf[j] <= prefixsum
        """
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
        # special case(the tail of the self.value is neutral_element(0))
        if idx == 2 * self.capacity - 1 and self.value[idx] == self.neutral_element:
            tmp = idx
            while tmp >= self.capacity and self.value[tmp] == self.neutral_element:
                tmp -= 1
            if tmp != self.capacity:
                idx = tmp
            else:
                raise ValueError("all element in tree are the neutral_element(0), can't find non-zero element")
        assert (self.value[idx] != self.neutral_element)
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity, operation=min)
