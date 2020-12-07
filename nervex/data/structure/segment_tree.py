import numpy as np


class SegmentTree:
    """
    Overview: segment tree, implemented by the tree-like array, only the leaf nodes are real value,
              the parents node is acquired by doing some operation on left and right child
    Interface: __init__, reduce, __setitem__, __getitem__
    """

    def __init__(self, capacity, operation, neutral_element=None):
        """
        Overview: initialize the segment tree
        Arguments:
            - capacity (:obj:`int`): the capacity of the tree (the number of the leaf nodes), should be the power of 2
            - operation (:obj:`function`): the operation function to construct the tree
            - neutral_element (:obj:`float` or None): the value of the neutral_element
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.operation = operation
        # set neutral/initial value for all the element
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
        # index 1 is the root, index capacity~2*capacity-1 are the leaf nodes
        # for each parent node with index i, left child is value[2*i] while right child is value[2*i+1]
        self.value = [self.neutral_element for _ in range(2 * capacity)]

    def reduce(self, start=0, end=None):
        """
        Overview: reduce the tree in range [start, end)
        Arguments:
            - start (:obj:`int`): start index(relative index, the first leaf node is 0)
            - end (:obj:`int` or None): end index(relative index)
        Returns:
            - reduce_result (:obj:`T`): the reduce result value, which is dependent on data type and operation
        """
        # TODO(nyz) check if directly reduce from the array(value) can be faster
        if end is None:
            end = self.capacity
        assert (start < end)

        # change to absolute leaf index
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
        """
        Overview: set leaf[idx] = val and update the related nodes
        Arguments:
            - idx (:obj:`int`): leaf node index
            - val (:obj:`T`): the value that will be assigned to leaf[idx]
        """
        assert (0 <= idx < self.capacity)
        idx += self.capacity
        self.value[idx] = val

        idx = idx >> 1  # transform to father node idx
        while idx >= 1:
            child_base = 2 * idx
            self.value[idx] = self.operation([self.value[child_base], self.value[child_base + 1]])
            idx = idx >> 1

    def __getitem__(self, idx):
        """
        Overview: get leaf[idx]
        Arguments:
            - idx (:obj:`int`): leaf node index
        Returns:
            - val (:obj:`T`): the value of leaf[idx]
        """
        assert (0 <= idx < self.capacity)
        return self.value[idx + self.capacity]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity, operation=sum)

    def find_prefixsum_idx(self, prefixsum, trust_caller=True):
        """
        Overview: find the highest non-zero index i, which for j in 0 <= j < i, sum_{j}leaf[j] <= prefixsum
        Arguments:
            - prefixsum (:obj:`T`): the target prefixsum
            - trust_caller (:obj:`bool`): whether to trust caller without check about prefixsum
        Returns:
            - idx (:obj:`int`): eligible index
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
