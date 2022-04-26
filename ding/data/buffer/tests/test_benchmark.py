import sys
import timeit
import torch
import pytest
import numpy as np

from ding.data.buffer import DequeBuffer
from ding.data.buffer.middleware import clone_object, PriorityExperienceReplay

# test different buffer size, eg: 1000, 10000, 100000;
size_list = [1000, 10000]
# test different tensor dim, eg: 32*32, 128*128, 512*512;
data_dim_list = [32, 128]
# repeat times.
repeats = 1000


class BufferBenchmark:

    def __init__(self, buffer_size, data_dim, buffer_type='base') -> None:
        self._buffer = DequeBuffer(size=buffer_size)
        if buffer_type == "clone":
            self._buffer.use(clone_object())
        if buffer_type == "priority":
            self._buffer.use(PriorityExperienceReplay(self._buffer, buffer_size=buffer_size, IS_weight=True))
        self._data = torch.rand(data_dim, data_dim)

    def data_storage(self) -> float:
        return sys.getsizeof(self._data.storage()) / 1024

    def count(self) -> int:
        return self._buffer.count()

    def push_op(self) -> None:
        self._buffer.push(self._data)

    def priority_push_op(self) -> None:
        self._buffer.push(self._data, meta={'priority': 2.0})

    def sample_op(self) -> None:
        self._buffer.sample(128, replace=False)

    def replace_sample_op(self) -> None:
        self._buffer.sample(128, replace=True)


def get_mean_std(res):
    return np.mean(res), np.std(res)


@pytest.mark.benchmark
@pytest.mark.parametrize('buffer_type', ['base', 'clone', 'priority'])
def test_clone_benchmark(buffer_type):
    for size in size_list:
        for dim in data_dim_list:
            assert size >= 128, "size is too small, please set an int no less than 128!"

            buffer_test = BufferBenchmark(size, dim, buffer_type)

            print("exp-buffer_{}_{}-data_{:.2f}_KB".format(buffer_type, size, buffer_test.data_storage()))

            # test pushing
            push_op = buffer_test.push_op
            if buffer_type == 'priority':
                push_op = buffer_test.priority_push_op
            mean, std = get_mean_std(timeit.repeat(push_op, number=repeats))
            print("Empty Push Test:         mean {:.4f} s, std {:.4f} s".format(mean, std))

            # fill the buffer before sampling tests
            for _ in range(size):
                buffer_test.push_op()
            assert buffer_test.count() == size, "buffer is not full when testing sampling!"

            # test sampling without replace
            mean, std = get_mean_std(timeit.repeat(buffer_test.sample_op, number=repeats))
            print("No-Replace Sample Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

            # test sampling with replace
            mean, std = get_mean_std(timeit.repeat(buffer_test.replace_sample_op, number=repeats))
            print("Replace Sample Test:     mean {:.4f} s, std {:.4f} s".format(mean, std))

            print("=" * 100)
