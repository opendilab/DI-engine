import random

import numpy as np
import pytest
import torch

from ding.envs.common.common_function import num_first_one_hot, sqrt_one_hot, div_one_hot, div_func, clip_one_hot, \
    reorder_one_hot, reorder_one_hot_array, reorder_boolean_vector, \
    get_to_and, batch_binary_encode, compute_denominator, get_postion_vector, \
    affine_transform

VALUES = [2, 3, 5, 7, 11]


@pytest.fixture(scope="function")
def setup_reorder_array():
    ret = np.full((12), -1)
    for i, v in enumerate(VALUES):
        ret[v] = i
    return ret


@pytest.fixture(scope="function")
def setup_reorder_dict():
    return {v: i for i, v in enumerate(VALUES)}


def generate_data():
    ret = {
        'obs': np.random.randn(4),
    }
    p_weight = np.random.uniform()
    if p_weight < 1. / 3:
        pass  # no key 'priority'
    elif p_weight < 2. / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform()

    return ret


@pytest.mark.unittest
class TestEnvCommonFunc:

    def test_one_hot(self):
        a = torch.Tensor([[3, 4, 5], [1, 2, 6]])

        a_sqrt = sqrt_one_hot(a, 6)
        assert a_sqrt.max().item() == 1
        assert [j.sum().item() for i in a_sqrt for j in i] == [1 for _ in range(6)]
        sqrt_dim = 3
        assert a_sqrt.shape == (2, 3, sqrt_dim)

        a_div = div_one_hot(a, 6, 2)
        assert a_div.max().item() == 1
        assert [j.sum().item() for i in a_div for j in i] == [1 for _ in range(6)]
        div_dim = 4
        assert a_div.shape == (2, 3, div_dim)

        a_di = div_func(a, 2)
        assert a_di.shape == (2, 1, 3)
        assert torch.eq(a_di.squeeze() * 2, a).all()

        a_clip = clip_one_hot(a.long(), 4)
        assert a_clip.max().item() == 1
        assert [j.sum().item() for i in a_clip for j in i] == [1 for _ in range(6)]
        clip_dim = 4
        assert a_clip.shape == (2, 3, clip_dim)

    def test_reorder(self, setup_reorder_array, setup_reorder_dict):
        a = torch.LongTensor([2, 7])  # VALUES = [2, 3, 5, 7, 11]

        a_array = reorder_one_hot_array(a, setup_reorder_array, 5)
        a_dict = reorder_one_hot(a, setup_reorder_dict, 5)
        assert torch.eq(a_array, a_dict).all()
        assert a_array.max().item() == 1
        assert [j.sum().item() for j in a_array] == [1 for _ in range(2)]
        reorder_dim = 5
        assert a_array.shape == (2, reorder_dim)

        a_bool = reorder_boolean_vector(a, setup_reorder_dict, 5)
        assert a_array.max().item() == 1
        assert torch.eq(a_bool, sum([_ for _ in a_array])).all()

    def test_binary(self):
        a = torch.LongTensor([445, 1023])
        a_binary = batch_binary_encode(a, 10)
        ans = []
        for number in a:
            one = [int(_) for _ in list(bin(number))[2:]]
            for _ in range(10 - len(one)):
                one.insert(0, 0)
            ans.append(one)
        ans = torch.Tensor(ans)
        assert torch.eq(a_binary, ans).all()

    def test_position(self):
        a = [random.randint(0, 5000) for _ in range(32)]
        a_position = get_postion_vector(a)
        assert a_position.shape == (64, )

    def test_affine_transform(self):
        a = torch.rand(4, 3)
        a = (a - a.min()) / (a.max() - a.min())
        a = a * 2 - 1
        ans = affine_transform(a, min_val=-2, max_val=2)
        assert ans.shape == (4, 3)
        assert ans.min() == -2 and ans.max() == 2
        a = np.random.rand(3, 5)
        a = (a - a.min()) / (a.max() - a.min())
        a = a * 2 - 1
        ans = affine_transform(a, alpha=4, beta=1)
        assert ans.shape == (3, 5)
        assert ans.min() == -3 and ans.max() == 5
