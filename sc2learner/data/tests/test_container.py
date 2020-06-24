import pytest
import torch
import numpy as np
import copy
from sc2learner.data.structure import SequenceContainer, SpecialContainer, TensorContainer

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


@pytest.mark.unittest
class TestSpecialContainer:
    def test_init(self):
        container = SpecialContainer(torch.randn(4))
        assert container.shape == (1, 1, 1)
        assert container._data_idx == 1

        container = SpecialContainer([torch.randn(4) for i in range(12)], shape=(2, 2, 3))
        assert container.shape == (2, 2, 3)
        assert container._data_idx == 12
        assert container._index_map['010'] == 4 - 1
        assert container._index_map['112'] == 12 - 1
        print(container)

    def test_cat(self):
        container = SpecialContainer([torch.randn(4) for i in range(12)], shape=(2, 2, 3))
        container_cat = SpecialContainer([torch.randn(4) for i in range(6)], shape=(1, 2, 3))
        assert container.shape == (2, 2, 3)
        assert container._data_idx == 12
        container.cat(container_cat, dim=0)
        assert container.shape == (3, 2, 3)
        assert container._data_idx == 12 + 6
        assert container._index_map['200'] == 12
        # error case
        with pytest.raises(AssertionError):
            container_cat = SpecialContainer([torch.randn(4) for i in range(8)], shape=(4, 2, 1))
            container.cat(container_cat, dim=0)

    def create_container(self):
        A, T, B = (3, 2, 4)
        tmp = []
        for _ in range(B):
            tmp.append(SpecialContainer(torch.randn(4)))
        for i in range(1, B):
            tmp[0].cat(tmp[i], dim=2)
        tmp = tmp[0]
        tmp_copy = copy.deepcopy(tmp)
        for _ in range(T - 1):
            tmp.cat(tmp_copy, dim=1)
        tmp_copy = copy.deepcopy(tmp)
        for _ in range(A - 1):
            tmp.cat(tmp_copy, dim=0)

        assert tmp.shape == (A, T, B)
        return tmp

    def test_getitem(self):
        container = self.create_container()
        # int
        assert container[1].shape == (1, 2, 4)
        # slice
        assert container[:].shape == (3, 2, 4)
        assert container[1:].shape == (2, 2, 4)
        assert container[1:2].shape == (1, 2, 4)
        # tuple
        assert container[:2, 1].shape == (2, 1, 4)
        assert container[:, :, 1].shape == (3, 2, 1)
        # dict
        assert container[{'agent_num': [0, 2]}].shape == (2, 2, 4)
        assert container[{'trajectory_len': [0], 'agent_num': [1]}].shape == (1, 1, 4)
        # error case
        with pytest.raises(AssertionError):
            container[3:2]
        with pytest.raises(AssertionError):
            container[-1:]
        with pytest.raises(AssertionError):
            container[5]

        # item
        data = container[0, 0, 0]
        assert data.shape == (1, 1, 1)
        data_item = data.item
        assert isinstance(data_item, torch.Tensor) and data_item.shape == (4, )


@pytest.mark.unittest
class TestTensorContainer:
    def test_init(self):
        container = TensorContainer(torch.randn(4, 5))
        assert container.shape == (1, 1, 1)
        assert container.item_shape == (4, 5)
        assert container.data.shape == (1, 1, 1, 4, 5)
        print(container)
        shape3 = (2, 3, 4)
        container = TensorContainer(torch.randn(*shape3, 5), shape=shape3)
        assert container.shape == shape3

        # error case
        with pytest.raises(AssertionError):
            container = TensorContainer(torch.randn(*shape3, 5), shape=[3] * 3)

    def test_cat(self):
        container = TensorContainer(torch.randn(4, 3))
        assert container.shape == (1, 1, 1)
        container.cat(container, dim=0)
        assert container.shape == (2, 1, 1)

        # error case
        container_cat_error = TensorContainer(torch.randn(4, 4))
        with pytest.raises(AssertionError):
            container.cat(container_cat_error, dim=0)

    def test_item(self):
        shape3 = (2, 3, 4)
        container = TensorContainer(torch.randn(*shape3), shape=shape3)
        assert container.shape == shape3
        item = container[0, 0, 0].item
        assert isinstance(item, torch.Tensor)
        assert item.shape == ()

        # error case
        with pytest.raises(AssertionError):
            container.item

    def test_getitem(self):
        shape3 = (2, 3, 4)
        container = TensorContainer(torch.randn(*shape3), shape=shape3)
        # int
        item = container[1]
        assert item.shape == (1, 3, 4)
        # slice
        item = container[:]
        assert item.shape == (2, 3, 4)
        item = container[1:]
        assert item.shape == (1, 3, 4)
        # tuple
        item = container[:, 1, :]
        assert item.shape == (2, 1, 4)
        item = container[1:, :, 0]
        assert item.shape == (1, 3, 1)
        # dict
        item = container[{'trajectory_len': [1, 2]}]
        assert item.shape == (2, 2, 4)
        item = container[{'agent_num': [1], 'batch_size': [0, 3]}]
        assert item.shape == (1, 3, 2)
