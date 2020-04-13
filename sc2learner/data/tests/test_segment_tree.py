import pytest
import numpy as np

from sc2learner.data.structure import SumSegmentTree, MinSegmentTree


@pytest.mark.unittest
class TestSumSegmentTree:
    def test_create(self):
        try:
            tree = SumSegmentTree(capacity=13)
            assert (False)
        except Exception as e:
            assert (isinstance(e, AssertionError))

        tree = SumSegmentTree(capacity=16)
        assert (tree.operation is sum)
        assert (tree.neutral_element == 0.)
        assert (max(tree.value) == 0.)
        assert (min(tree.value) == 0.)

    def test_set_get_item(self):
        tree = SumSegmentTree(capacity=4)
        elements = [1, 5, 4, 7]
        get_result = []
        for idx, val in enumerate(elements):
            tree[idx] = val
            get_result.append(tree[idx])

        assert (elements == get_result)
        assert (tree.reduce() == sum(elements))
        assert (tree.reduce(0, 3) == sum(elements[:3]))
        assert (tree.reduce(0, 2) == sum(elements[:2]))
        assert (tree.reduce(0, 1) == sum(elements[:1]))
        assert (tree.reduce(1, 3) == sum(elements[1:3]))
        assert (tree.reduce(1, 2) == sum(elements[1:2]))
        assert (tree.reduce(2, 3) == sum(elements[2:3]))

        try:
            tree.reduce(2, 2)
            assert (False)
        except Exception as e:
            assert (isinstance(e, AssertionError))

    def test_find_prefixsum_idx(self):
        tree = SumSegmentTree(capacity=8)
        elements = [0, 0.1, 0.5, 0, 0, 0.2, 0.8, 0]
        for idx, val in enumerate(elements):
            tree[idx] = val
        try:
            tree.find_prefixsum_idx(tree.reduce() + 1e-4, trust_caller=False)
            assert False
        except Exception as e:
            assert (isinstance(e, AssertionError))
        try:
            tree.find_prefixsum_idx(-1e-6, trust_caller=False)
            assert False
        except Exception as e:
            assert (isinstance(e, AssertionError))

        assert (tree.find_prefixsum_idx(0) == 1)
        assert (tree.find_prefixsum_idx(0.09) == 1)
        assert (tree.find_prefixsum_idx(0.1) == 2)
        assert (tree.find_prefixsum_idx(0.59) == 2)
        assert (tree.find_prefixsum_idx(0.6) == 5)
        assert (tree.find_prefixsum_idx(0.799) == 5)
        assert (tree.find_prefixsum_idx(0.8) == 6)
        assert (tree.find_prefixsum_idx(tree.reduce()) == 6)


class TestMinSegmentTree:
    def test_create(self):
        tree = MinSegmentTree(capacity=16)
        assert (tree.operation is min)
        assert (tree.neutral_element == np.inf)
        assert (max(tree.value) == np.inf)
        assert (min(tree.value) == np.inf)

    def test_set_get_item(self):
        tree = MinSegmentTree(capacity=4)
        elements = [1, -10, 10, 7]
        get_result = []
        for idx, val in enumerate(elements):
            tree[idx] = val
            get_result.append(tree[idx])

        assert (elements == get_result)
        assert (tree.reduce() == min(elements))
        assert (tree.reduce(0, 3) == min(elements[:3]))
        assert (tree.reduce(0, 2) == min(elements[:2]))
        assert (tree.reduce(0, 1) == min(elements[:1]))
        assert (tree.reduce(1, 3) == min(elements[1:3]))
        assert (tree.reduce(1, 2) == min(elements[1:2]))
        assert (tree.reduce(2, 3) == min(elements[2:3]))
