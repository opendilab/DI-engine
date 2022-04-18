import sys
import timeit
import torch
import pytest
import numpy as np

# test different buffer size, eg: 1000, 10000, 100000;
size_list = [1000, 10000, 100000]
# test different tensor dim, eg: 32*32, 128*128, 512*512;
data_dim_list = [32, 128]
# repeat times.
repeats = 100

empty_base_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
buffer = DequeBuffer(size={0})
data = torch.rand({1}, {1})
"""

full_base_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
buffer = DequeBuffer(size={0})
for i in range({0}):
    data = torch.rand({1}, {1})
    buffer.push(data)
"""

empty_clone_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import clone_object
buffer = DequeBuffer(size={0}).use(clone_object())
data = torch.rand({1}, {1})
"""

full_clone_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import clone_object
buffer = DequeBuffer(size={0}).use(clone_object())
for i in range({0}):
    data = torch.rand({1}, {1})
    buffer.push(data)
"""

empty_priority_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import PriorityExperienceReplay
buffer = DequeBuffer(size={0})
buffer.use(PriorityExperienceReplay(buffer, buffer_size={0}, IS_weight=True))
data = torch.rand({1}, {1})
"""

full_priority_setup = """import sys
import torch
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import PriorityExperienceReplay
buffer = DequeBuffer(size={0})
buffer.use(PriorityExperienceReplay(buffer, buffer_size={0}, IS_weight=True))
for i in range({0}):
    data = torch.rand({1}, {1})
    buffer.push(data, meta=dict(priority=2.0))
"""
print(full_priority_setup.format(100, 16))


def get_mean_std(res):
    return np.mean(res), np.std(res)


def _timeit(buf_type, buf_size, data_dim):
    # initialize setup and target statements
    empty_setup = eval("empty_{}_setup".format(buf_type))
    full_setup = eval("full_{}_setup".format(buf_type))

    push_stmt = "buffer.push(data)"
    if buf_type == "priority":
        push_stmt = "buffer.push(data, meta=dict(priority=2.0))"
    sample_stmt = "buffer.sample(128, replace=False)"
    replace_sample_stmt = "buffer.sample(128, replace=True)"

    print(
        "exp-buffer_{}_{}-data_{:.2f}_KB".format(
            buf_type, buf_size,
            sys.getsizeof(torch.rand(data_dim, data_dim).storage()) / 1024
        )
    )

    # test pushing
    mean, std = get_mean_std(timeit.repeat(push_stmt, setup=empty_setup.format(buf_size, data_dim), number=repeats))
    print("Empty Push Test:          mean {:.4f} s, std {:.4f} s".format(mean, std))

    # test sampling without replace
    mean, std = get_mean_std(timeit.repeat(sample_stmt, setup=full_setup.format(buf_size, data_dim), number=repeats))
    print("No-Replace Sample Test:   mean {:.4f} s, std {:.4f} s".format(mean, std))

    # test sampling with replace
    mean, std = get_mean_std(
        timeit.repeat(replace_sample_stmt, setup=full_setup.format(buf_size, data_dim), number=repeats)
    )
    print("Replace Sample Test:      mean {:.4f} s, std {:.4f} s".format(mean, std))

    # Attention:
    # the test results are the sum of repeats, rather than avg!

    print("=" * 100)


@pytest.mark.benchmark
def test_base_benchmark():
    for size in size_list:
        for dim in data_dim_list:
            _timeit("base", size, dim)


@pytest.mark.benchmark
def test_clone_benchmark():
    for size in size_list:
        for dim in data_dim_list:
            _timeit("clone", size, dim)


@pytest.mark.benchmark
def test_priority_benchmark():
    for size in size_list:
        for dim in data_dim_list:
            _timeit("priority", size, dim)
