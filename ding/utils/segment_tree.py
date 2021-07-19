from functools import partial, lru_cache
from typing import Callable, Optional

import numpy as np

import ding
from .default_helper import one_time_warning


@lru_cache()
def njit():
    try:
        if ding.enable_numba:
            import numba
            from numba import njit as _njit
            version = numba.__version__
            middle_version = version.split(".")[1]
            if int(middle_version) < 53:
                _njit = partial  # noqa
                one_time_warning(
                    "Due to your numba version <= 0.53.0, DI-engine disables it. And you can install \
                    numba==0.53.0 if you want to speed up something"
                )
        else:
            _njit = partial
    except ImportError:
        one_time_warning("If you want to use numba to speed up segment tree, please install numba first")
        _njit = partial
    return _njit


class SegmentTree:
    """
    Overview:
        Segment tree data structure, implemented by the tree-like array. Only the leaf nodes are real value,
        non-leaf nodes are to do some operations on its left and right child.
    Interface:
        ``__init__``, ``reduce``, ``__setitem__``, ``__getitem__``
    """

    def __init__(self, capacity: int, operation: Callable, neutral_element: Optional[float] = None) -> None:
        """
        Overview:
            Initialize the segment tree. Tree's root node is at index 1.
        Arguments:
            - capacity (:obj:`int`): Capacity of the tree (the number of the leaf nodes), should be the power of 2.
            - operation (:obj:`function`): The operation function to construct the tree, e.g. sum, max, min, etc.
            - neutral_element (:obj:`float` or :obj:`None`): The value of the neutral element, which is used to init \
                all nodes value in the tree.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.operation = operation
        # Set neutral value(initial value) for all elements.
        if neutral_element is None:
            if operation == 'sum':
                neutral_element = 0.
            elif operation == 'min':
                neutral_element = np.inf
            elif operation == 'max':
                neutral_element = -np.inf
            else:
                raise ValueError("operation argument should be in min, max, sum (built in python functions).")
        self.neutral_element = neutral_element
        # Index 1 is the root; Index ranging in [capacity, 2 * capacity - 1] are the leaf nodes.
        # For each parent node with index i, left child is value[2*i] and right child is value[2*i+1].
        self.value = np.full([capacity * 2], neutral_element)
        self._compile()

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Overview:
            Reduce the tree in range ``[start, end)``
        Arguments:
            - start (:obj:`int`): Start index(relative index, the first leaf node is 0), default set to 0
            - end (:obj:`int` or :obj:`None`): End index(relative index), default set to ``self.capacity``
        Returns:
            - reduce_result (:obj:`float`): The reduce result value, which is dependent on data type and operation
        """
        # TODO(nyz) check if directly reduce from the array(value) can be faster
        if end is None:
            end = self.capacity
        assert (start < end)
        # Change to absolute leaf index by adding capacity.
        start += self.capacity
        end += self.capacity
        return _reduce(self.value, start, end, self.neutral_element, self.operation)

    def __setitem__(self, idx: int, val: float) -> None:
        """
        Overview:
            Set ``leaf[idx] = val``; Then update the related nodes.
        Arguments:
            - idx (:obj:`int`): Leaf node index(relative index), should add ``capacity`` to change to absolute index.
            - val (:obj:`float`): The value that will be assigned to ``leaf[idx]``.
        """
        assert (0 <= idx < self.capacity), idx
        # ``idx`` should add ``capacity`` to change to absolute index.
        _setitem(self.value, idx + self.capacity, val, self.operation)

    def __getitem__(self, idx: int) -> float:
        """
        Overview:
            Get ``leaf[idx]``
        Arguments:
            - idx (:obj:`int`): Leaf node ``index(relative index)``, add ``capacity`` to change to absolute index.
        Returns:
            - val (:obj:`float`): The value of ``leaf[idx]``
        """
        assert (0 <= idx < self.capacity)
        return self.value[idx + self.capacity]

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        for d in [f64, f32, i64]:
            _setitem(d, 0, 3.0, 'sum')
            _reduce(d, 0, 1, 0.0, 'min')
            _find_prefixsum_idx(d, 1, 0.5, 0.0)


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity: int) -> None:
        """
        Overview:
            Init sum segment tree by passing ``operation='sum'``
        """
        super(SumSegmentTree, self).__init__(capacity, operation='sum')

    def find_prefixsum_idx(self, prefixsum: float, trust_caller: bool = True) -> int:
        """
        Overview:
            Find the highest non-zero index i, sum_{j}leaf[j] <= ``prefixsum`` (where 0 <= j < i)
            and sum_{j}leaf[j] > ``prefixsum`` (where 0 <= j < i+1)
        Arguments:
            - prefixsum (:obj:`float`): The target prefixsum.
            - trust_caller (:obj:`bool`): Whether to trust caller, which means whether to check whether \
                this tree's sum is greater than the input ``prefixsum`` by calling ``reduce`` function.
                Default set to True.
        Returns:
            - idx (:obj:`int`): Eligible index.
        """
        if not trust_caller:
            assert 0 <= prefixsum <= self.reduce() + 1e-5, prefixsum
        return _find_prefixsum_idx(self.value, self.capacity, prefixsum, self.neutral_element)


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity: int) -> None:
        """
        Overview:
            Init sum segment tree by passing ``operation='min'``
        """
        super(MinSegmentTree, self).__init__(capacity, operation='min')


@njit()
def _setitem(tree: np.ndarray, idx: int, val: float, operation: str) -> None:
    tree[idx] = val
    # Update from specified node to the root node
    while idx > 1:
        idx = idx >> 1  # To parent node idx
        left, right = tree[2 * idx], tree[2 * idx + 1]
        if operation == 'sum':
            tree[idx] = left + right
        elif operation == 'min':
            tree[idx] = min([left, right])


@njit()
def _reduce(tree: np.ndarray, start: int, end: int, neutral_element: float, operation: str) -> float:
    # Nodes in 【start, end) will be aggregated
    result = neutral_element
    while start < end:
        if start & 1:
            # If current start node (tree[start]) is a right child node, operate on start node and increase start by 1
            if operation == 'sum':
                result = result + tree[start]
            elif operation == 'min':
                result = min([result, tree[start]])
            start += 1
        if end & 1:
            # If current end node (tree[end - 1]) is right child node, decrease end by 1 and operate on end node
            end -= 1
            if operation == 'sum':
                result = result + tree[end]
            elif operation == 'min':
                result = min([result, tree[end]])
        # Both start and end transform to respective parent node
        start = start >> 1
        end = end >> 1
    return result


@njit()
def _find_prefixsum_idx(tree: np.ndarray, capacity: int, prefixsum: float, neutral_element: float) -> int:
    # The function is to find a non-leaf node's index which satisfies:
    # self.value[idx] > input prefixsum and self.value[idx + 1] <= input prefixsum
    # In other words, we can assume that there are intervals: [num_0, num_1), [num_1, num_2), ... [num_k, num_k+1),
    # the function is to find input prefixsum falls in which interval and return the interval's index.
    idx = 1  # start from root node
    while idx < capacity:
        child_base = 2 * idx
        if tree[child_base] > prefixsum:
            idx = child_base
        else:
            prefixsum -= tree[child_base]
            idx = child_base + 1
    # Special case: The last element of ``self.value`` is neutral_element(0),
    # and caller wants to ``find_prefixsum_idx(root_value)``.
    # However, input prefixsum should be smaller than root_value.
    if idx == 2 * capacity - 1 and tree[idx] == neutral_element:
        tmp = idx
        while tmp >= capacity and tree[tmp] == neutral_element:
            tmp -= 1
        if tmp != capacity:
            idx = tmp
        else:
            raise ValueError("All elements in tree are the neutral_element(0), can't find non-zero element")
    assert (tree[idx] != neutral_element)
    return idx - capacity
