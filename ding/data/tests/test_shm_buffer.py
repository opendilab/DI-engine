import pytest
import numpy as np
import timeit
from ding.data.shm_buffer import ShmBuffer
import multiprocessing as mp


def subprocess(shm_buf):
    data = np.random.rand(1024, 1024).astype(np.float32)
    res = timeit.repeat(lambda: shm_buf.fill(data), repeat=5, number=1000)
    print("Mean: {:.4f}s, STD: {:.4f}s, Mean each call: {:.4f}ms".format(np.mean(res), np.std(res), np.mean(res)))


@pytest.mark.benchmark
def test_shm_buffer():
    data = np.random.rand(1024, 1024).astype(np.float32)
    shm_buf = ShmBuffer(data.dtype, data.shape, copy_on_get=False)
    proc = mp.Process(target=subprocess, args=[shm_buf])
    proc.start()
    proc.join()
