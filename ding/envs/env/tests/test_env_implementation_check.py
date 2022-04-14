import pytest
from easydict import EasyDict
import numpy as np
import gym
from copy import deepcopy

from ding.envs.env import check_array_space, check_different_memory, check_all, demonstrate_correct_procedure
from ding.envs.env.tests import DemoEnv


@pytest.mark.unittest
def test_an_implemented_env():
    demo_env = DemoEnv({})
    check_all(demo_env)
    demonstrate_correct_procedure(DemoEnv)


@pytest.mark.unittest
def test_check_array_space():
    seq_array = (np.array([1, 2, 3], dtype=np.int64), np.array([4., 5., 6.], dtype=np.float32))
    seq_space = [gym.spaces.Box(low=0, high=10, shape=(3, ), dtype=np.int64) for _ in range(2)]
    with pytest.raises(AssertionError):
        check_array_space(seq_array, seq_space, 'test_sequence')

    dict_array = {'a': np.array([1, 2, 3], dtype=np.int64), 'b': np.array([4., 5., 6.], dtype=np.float32)}
    int_box = gym.spaces.Box(low=0, high=10, shape=(3, ), dtype=np.int64)
    dict_space = {'a': deepcopy(int_box), 'b': deepcopy(int_box)}
    with pytest.raises(AssertionError):
        check_array_space(dict_array, dict_space, 'test_dict')

    with pytest.raises(TypeError):
        check_array_space(1, dict_space, 'test_type_error')


@pytest.mark.unittest
def test_check_different_memory():
    int_seq = np.array([1, 2, 3], dtype=np.int64)
    seq_array1 = (int_seq, np.array([4., 5., 6.], dtype=np.float32))
    seq_array2 = (int_seq, np.array([4., 5., 6.], dtype=np.float32))
    with pytest.raises(AssertionError):
        check_different_memory(seq_array1, seq_array2, -1)

    dict_array1 = {'a': np.array([4., 5., 6.], dtype=np.float32), 'b': int_seq}
    dict_array2 = {'a': np.array([4., 5., 6.], dtype=np.float32), 'b': int_seq}
    with pytest.raises(AssertionError):
        check_different_memory(dict_array1, dict_array2, -1)

    with pytest.raises(AssertionError):
        check_different_memory(1, dict_array1, -1)
    with pytest.raises(TypeError):
        check_different_memory(1, 2, -1)
