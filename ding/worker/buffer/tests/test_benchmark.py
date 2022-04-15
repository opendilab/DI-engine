import pytest
import numpy as np
import time
import sys
import torch
import timeit


buffer_list = ["base", "clone", "priority"]
size_list = [1000, 10000, 100000]
data_dim_list = [32, 128, 512]
n = 10000


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
data = torch.rand({1}, {1})
for i in range({0}):
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
data = torch.rand({1}, {1})
for i in range({0}):
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
data = torch.rand({1}, {1})
for i in range({0}):
    buffer.push(data)
"""


def get_mean_std(res):
    return np.mean(res), np.std(res)


@pytest.mark.benchmark
def test_buffer_benchmark():
    for buf in buffer_list:
        for s in size_list:
            for d in data_dim_list:
                empty_setup = eval("empty_{}_setup".format(buf))
                full_setup = eval("full_{}_setup".format(buf))

                push_stmt = "buffer.push(data)"
                if buf == "priority":
                    push_stmt = "buffer.push(data, meta={'priority': 2.0})"
                sample_stmt = "buffer.sample(128, replace=False)"
                replace_sample_stmt = "buffer.sample(128, replace=True)"

                print("exp-buffer_{}-data_{:.2f}_KB".format(s, sys.getsizeof(torch.rand(d, d).storage()) / 1024))

                mean, std = get_mean_std(timeit.repeat(push_stmt, setup=empty_setup.format(s, d)))
                print("Empty Push Test:          mean {:.4f} s, std {:.4f} s".format(mean, std))

                mean, std = get_mean_std(timeit.repeat(push_stmt, setup=full_setup.format(s, d)))
                print("Full Push Test:           mean {:.4f} s, std {:.4f} s".format(mean, std))

                mean, std = get_mean_std(timeit.repeat(sample_stmt, setup=full_setup.format(s, d)))
                print("No-Replace Sample Test:   mean {:.4f} s, std {:.4f} s".format(mean, std))

                mean, std = get_mean_std(timeit.repeat(replace_sample_stmt, setup=full_setup.format(s, d)))
                print("Replace Sample Test:      mean {:.4f} s, std {:.4f} s".format(mean, std))


@pytest.mark.benchmark
def test_buffer_benchmark():
    for buf in buffer_list:
        for s in size_list:
            for d in data_dim_list:
                empty_setup = eval("empty_{}_setup".format(buf))
                full_setup = eval("full_{}_setup".format(buf))

                push_stmt = "buffer.push(data)"
                if buf == "priority":
                    push_stmt = "buffer.push(data, meta={'priority': 2.0})"
                sample_stmt = "buffer.sample(128, replace=False)"
                replace_sample_stmt = "buffer.sample(128, replace=True)"

                print("exp-buffer_{}_{}-data_{:.2f}_KB".format(buf, s, sys.getsizeof(torch.rand(d, d).storage()) / 1024))

                mean, std = get_mean_std(timeit.repeat(push_stmt, setup=empty_setup.format(s, d), number=n))
                print("Empty Push Test:          mean {:.4f} s, std {:.4f} s".format(mean, std))
                time.sleep(1.0)

                mean, std = get_mean_std(timeit.repeat(push_stmt, setup=full_setup.format(s, d), number=n))
                print("Full Push Test:           mean {:.4f} s, std {:.4f} s".format(mean, std))
                time.sleep(1.0)

                mean, std = get_mean_std(timeit.repeat(sample_stmt, setup=full_setup.format(s, d), number=n))
                print("No-Replace Sample Test:   mean {:.4f} s, std {:.4f} s".format(mean, std))
                time.sleep(1.0)

                mean, std = get_mean_std(timeit.repeat(replace_sample_stmt, setup=full_setup.format(s, d), number=n))
                print("Replace Sample Test:      mean {:.4f} s, std {:.4f} s".format(mean, std))
                time.sleep(1.0)

                print("=" * 100)
