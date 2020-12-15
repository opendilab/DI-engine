import numpy as np
import warnings
from functools import partial
from typing import Callable, Optional, Union, Any
try:
    from numba import njit, jit
except ImportError:
    warnings.warn("If you want to use numba to speed up segment tree, please install numba first")
    njit = partial

op2str = {sum: 'sum', min: 'min'}


class SegmentTree:
    """
    Overview:
        Segment tree data structure, implemented by the tree-like array. Only the leaf nodes are real value,
        the parent node is acquired by doing some operation on left and right child
    Interface:
        __init__, reduce, __setitem__, __getitem__
    """

    def __init__(self, capacity: int, operation: Callable, neutral_element: Optional[float] = None) -> None:
        """
        Overview:
            initialize the segment tree
        Arguments:
            - capacity (:obj:`int`): the capacity of the tree (the number of the leaf nodes), should be the power of 2
            - operation (:obj:`function`): the operation function to construct the tree, e.g. sum, max, min
            - neutral_element (:obj:`float` or :obj:`None`): the value of the neutral element, which is used to init \
                all nodes value in the tree
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
        # index 1 is the root, index ranging in [capacity, 2 * capacity - 1] are the leaf nodes
        # for each parent node with index i, left child is value[2*i] while right child is value[2*i+1]
        self.value = np.full([capacity * 2], neutral_element)
        # self.value = [neutral_element for _ in range(capacity * 2)]
        self._compile()

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Overview:
            Reduce the tree in range [start, end)
        Arguments:
            - start (:obj:`int`): start index(relative index, the first leaf node is 0), default set to 0
            - end (:obj:`int` or :obj:`None`): end index(relative index), default set to ``self.capacity``
        Returns:
            - reduce_result (:obj:`float`): the reduce result value, which is dependent on data type and operation
        """
        # TODO(nyz) check if directly reduce from the array(value) can be faster
        if end is None:
            end = self.capacity
        assert (start < end)
        # change to absolute leaf index by adding capacitty
        start += self.capacity
        end += self.capacity
        return _reduce(self.value, start, end, self.neutral_element, op2str[self.operation])
        # return _reduce(self.value, start, end, self.neutral_element, self.operation)

    def __setitem__(self, idx: int, val: float) -> None:
        """
        Overview:
            Set leaf[idx] = val; Then update the related nodes
        Arguments:
            - idx (:obj:`int`): leaf node index(relative index), should add ``capacity`` and change to absolute index
            - val (:obj:`float`): the value that will be assigned to leaf[idx]
        """
        assert (0 <= idx < self.capacity)
        idx += self.capacity
        _setitem(self.value, idx, val, op2str[self.operation])
        # _setitem(self.value, idx, val, self.operation)

    def __getitem__(self, idx: int) -> float:
        """
        Overview:
            Get leaf[idx]
        Arguments:
            - idx (:obj:`int`): leaf node index(relative index), should add ``capacity`` and change to absolute index
        Returns:
            - val (:obj:`float`): the value of leaf[idx]
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
        super(SumSegmentTree, self).__init__(capacity, operation=sum)

    def find_prefixsum_idx(self, prefixsum: float, trust_caller: bool = True) -> int:
        """
        Overview:
            Find the highest non-zero index i, which for j in 0 <= j < i, sum_{j}leaf[j] <= prefixsum
        Arguments:
            - prefixsum (:obj:`float`): the target prefixsum
            - trust_caller (:obj:`bool`): whether to trust caller, which means not checking whether this tree \
                satisfies the prefixsum by calling ``reduce`` function, default set to True
        Returns:
            - idx (:obj:`int`): eligible index
        """
        if not trust_caller:
            assert 0 <= prefixsum <= self.reduce() + 1e-5
        # find a non-leaf node's index which satisfies self.value[idx] > original prefixsum
        return _find_prefixsum_idx(self.value, self.capacity, prefixsum, self.neutral_element)


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity: int) -> None:
        super(MinSegmentTree, self).__init__(capacity, operation=min)


@njit
def _setitem(tree: np.ndarray, idx: int, val: float, operation: str) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[idx] = val
    # update from specified node to the root
    while idx > 1:
        idx = idx >> 1  # to parent node idx
        left, right = tree[2 * idx], tree[2 * idx + 1]
        # print('----setitem----', left, type(left), right, type(right))
        if operation == 'sum':
            # tree[idx] = np.sum([left.item(), right.item()], dtype=np.float64)
            tree[idx] = left + right
        elif operation == 'min':
            # tree[idx] = np.min([left.item(), right.item()], dtype=np.float64)
            tree[idx] = min([left, right])
        # tree[idx] = operation([tree[2 * idx], tree[2 * idx + 1]])
        # tree[idx] = getattr(np, operation)([tree[2 * idx], tree[2 * idx + 1]])


@njit
def _reduce(tree: np.ndarray, start: int, end: int, neutral_element: float, operation: str) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""
    # nodes in (start, end) should be aggregated
    result = neutral_element
    while start < end:
        if start & 1:
            # if current start node (tree[start]) is right child node, operate on start node and increas start by 1
            if operation == 'sum':
                # result = np.sum([result, tree[start]])
                result = result + tree[start]
            elif operation == 'min':
                # result = np.min([result, tree[start]])
                result = min([result, tree[start]])
            # result = operation([result, tree[start]])
            # result = getattr(np, operation)([result, tree[start]])
            start += 1
        if end & 1:
            # if current end node (tree[end - 1]) is right child node, decrease end by 1 and operate on end node
            end -= 1
            if operation == 'sum':
                # result = np.sum([result, tree[end]])
                result = result + tree[end]
            elif operation == 'min':
                # result = np.min([result, tree[end]])
                result = min([result, tree[end]])
            # result = operation([result, tree[end]])
            # result = getattr(np, operation)([result, tree[end]])

        # both transform to parent node
        start = start >> 1
        end = end >> 1
    return result


@njit
def _find_prefixsum_idx(tree: np.ndarray, capacity: int, prefixsum: float, neutral_element: float) -> int:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.

    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    # find a non-leaf node's index which satisfies self.value[idx] > original prefixsum
    idx = 1  # start from root node
    while idx < capacity:
        child_base = 2 * idx
        if tree[child_base] > prefixsum:
            idx = child_base
        else:
            prefixsum -= tree[child_base]
            idx = child_base + 1
    # special case(the last element of the ``self.value``` is neutral_element(0))
    if idx == 2 * capacity - 1 and tree[idx] == neutral_element:
        tmp = idx
        while tmp >= capacity and tree[tmp] == neutral_element:
            tmp -= 1
        if tmp != capacity:
            idx = tmp
        else:
            raise ValueError("all elements in tree are the neutral_element(0), can't find non-zero element")
    assert (tree[idx] != neutral_element)
    return idx - capacity
