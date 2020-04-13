import pytest
import torch
import numpy as np
import copy
from sc2learner.data.structure import SequenceContainer

data_generation_func = {'obs': lambda: torch.randn(1, 6), 'reward': lambda: [np.random.randint(0, 2)]}


def generate_data():
    ret = {}
    for k, f in data_generation_func.items():
        ret[k] = f()
    return ret


@pytest.mark.unittest
class TestSequenceContainer:
    def test_create(self):
        container = SequenceContainer(**generate_data())
        assert (container.name == 'SequenceContainer')
        assert (len(container) == 1)
        assert (container.keys == list(data_generation_func.keys()))
        assert (isinstance(container.value('obs'), torch.Tensor))
        assert (isinstance(container.value('reward'), list))
        try:
            container.value('xxx')
            assert False
        except AssertionError as e:
            assert True

    def test_cat(self):
        N = 10
        containers = [SequenceContainer(**generate_data()) for _ in range(N)]
        cat_container = copy.deepcopy(containers[0])
        for i in range(1, N):
            cat_container.cat(containers[i])
        assert (cat_container.value('obs').shape[0] == N)
        assert (len(cat_container.value('reward')) == N)
        assert (len(cat_container) == N)
        assert (cat_container.keys == list(data_generation_func.keys()))

    def test_getitem(self):
        N = 10
        containers = [SequenceContainer(**generate_data()) for _ in range(N)]
        cat_container = copy.deepcopy(containers[0])
        if len(cat_container) == 1:
            assert (cat_container == cat_container[0])
        for i in range(1, N):
            cat_container.cat(containers[i])
        assert (cat_container[0] == containers[0])
        assert (cat_container[1] == containers[1])
        assert (cat_container[N - 1] == containers[N - 1])
