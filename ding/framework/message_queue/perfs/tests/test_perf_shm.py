from ding.framework.message_queue.perfs.perf_shm import shm_perf_main
import multiprocessing as mp
import pytest
import torch


@pytest.mark.mqbenchmark
@pytest.mark.cudatest
@pytest.mark.multiprocesstest
def test_shm_numpy_shm():
    if torch.cuda.is_available():
        shm_perf_main("shm")


@pytest.mark.mqbenchmark
@pytest.mark.cudatest
@pytest.mark.multiprocesstest
def test_shm_cuda_shared_tensor():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        shm_perf_main("cuda_ipc")
