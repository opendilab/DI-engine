import sys
import timeit
import torch
import random
import pytest
import numpy as np

from ding.data.buffer import DequeBuffer
from ding.data.buffer.middleware import clone_object, PriorityExperienceReplay

# test different buffer size, eg: 1000, 10000, 100000;
size_list = [1000, 10000]
# test different tensor dim, eg: 32*32, 128*128, 512*512;
data_dim_list = [32, 128]
# repeat times.
repeats = 100


class BufferBenchmark:

    def __init__(self, buffer_size, data_dim, buffer_type='base') -> None:
        self._buffer = DequeBuffer(size=buffer_size)
        self._meta = dict()
        if buffer_type == "clone":
            self._buffer.use(clone_object())
        if buffer_type == "priority":
            self._buffer.use(PriorityExperienceReplay(self._buffer))
            self._meta["priority"] = 2.0
        self._data = {"obs": torch.rand(data_dim, data_dim)}

    def data_storage(self) -> float:
        return sys.getsizeof(self._data["obs"].storage()) / 1024

    def count(self) -> int:
        return self._buffer.count()

    def push_op(self) -> None:
        self._buffer.push(self._data, meta=self._meta)

    def push_with_group_info(self, num_keys=256) -> None:
        meta = self._meta.copy()
        rand = random.random()
        value = int(rand * num_keys)
        meta['group'] = value
        self._buffer.push(self._data, meta=meta)

    def sample_op(self) -> None:
        self._buffer.sample(128, replace=False)

    def replace_sample_op(self) -> None:
        self._buffer.sample(128, replace=True)

    def groupby_sample_op(self) -> None:
        self._buffer.sample(128, groupby="group")


def get_mean_std(res):
    # return the total time per 1000 ops
    return np.mean(res) * 1000.0 / repeats, np.std(res) * 1000.0 / repeats


@pytest.mark.benchmark
@pytest.mark.parametrize('buffer_type', ['base', 'clone', 'priority'])
def test_benchmark(buffer_type):
    for size in size_list:
        for dim in data_dim_list:
            assert size >= 128, "size is too small, please set an int no less than 128!"

            buffer_test = BufferBenchmark(size, dim, buffer_type)

            print("exp-buffer_{}_{}-data_{:.2f}_KB".format(buffer_type, size, buffer_test.data_storage()))

            # test pushing
            mean, std = get_mean_std(timeit.repeat(buffer_test.push_op, number=repeats))
            print("Empty Push Test:         mean {:.4f} s, std {:.4f} s".format(mean, std))

            # fill the buffer before sampling tests
            for _ in range(size):
                buffer_test.push_with_group_info()
            assert buffer_test.count() == size, "buffer is not full when testing sampling!"

            # test sampling without replace
            mean, std = get_mean_std(timeit.repeat(buffer_test.sample_op, number=repeats))
            print("No-Replace Sample Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

            # test sampling with replace
            mean, std = get_mean_std(timeit.repeat(buffer_test.replace_sample_op, number=repeats))
            print("Replace Sample Test:     mean {:.4f} s, std {:.4f} s".format(mean, std))

            # test groupby sampling
            if buffer_type != 'priority':
                mean, std = get_mean_std(timeit.repeat(buffer_test.groupby_sample_op, number=repeats))
                print("Groupby Sample Test:  mean {:.4f} s, std {:.4f} s".format(mean, std))

            print("=" * 100)
