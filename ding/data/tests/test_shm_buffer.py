from ding.data.shm_buffer import ShmBuffer, ShmBufferCuda
from ding.compatibility import torch_ge_1121

import pytest
import numpy as np
import timeit
import torch
import time


def subprocess_np_shm(shm_buf):
    data = np.random.rand(1024, 1024).astype(np.float32)
    res = timeit.repeat(lambda: shm_buf.fill(data), repeat=5, number=1000)
    print("Mean: {:.4f}s, STD: {:.4f}s, Mean each call: {:.4f}ms".format(np.mean(res), np.std(res), np.mean(res)))


def subprocess_cuda_shared_tensor(shm_buf_np, shm_buf_torch, event_run):
    event_run.wait()
    rtensor = shm_buf_torch.get()
    assert isinstance(rtensor, torch.Tensor)
    assert rtensor.device == torch.device('cuda:0')
    assert rtensor.dtype == torch.float32
    assert rtensor.sum().item() == 1024 * 1024

    rarray = shm_buf_np.get()
    assert isinstance(rarray, np.ndarray)
    assert rarray.dtype == np.dtype(np.float32)
    assert rarray.dtype == np.dtype(np.float32)

    res = timeit.repeat(lambda shm_buf_torch=shm_buf_torch: shm_buf_torch.get(), repeat=5, number=1000)
    print("CUDA-shared-tensor (torch) Get: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))
    res = timeit.repeat(lambda shm_buf_np=shm_buf_np: shm_buf_np.get(), repeat=5, number=1000)
    print("CUDA-shared-tensor (numpy) Get: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))

    del shm_buf_np
    del shm_buf_torch


@pytest.mark.benchmark
def test_shm_buffer():
    import multiprocessing as mp
    data = np.random.rand(1024, 1024).astype(np.float32)
    shm_buf = ShmBuffer(data.dtype, data.shape, copy_on_get=False)
    proc = mp.Process(target=subprocess_np_shm, args=[shm_buf])
    proc.start()
    proc.join()


@pytest.mark.benchmark
@pytest.mark.unittest
@pytest.mark.cudatest
def test_cuda_shm():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')

        event_run = ctx.Event()
        shm_buf_np = ShmBufferCuda(np.dtype(np.float32), shape=(1024, 1024), copy_on_get=True)
        shm_buf_torch = ShmBufferCuda(torch.float32, shape=(1024, 1024), copy_on_get=True)
        proc = ctx.Process(target=subprocess_cuda_shared_tensor, args=[shm_buf_np, shm_buf_torch, event_run])
        proc.start()

        ltensor = torch.ones((1024, 1024), dtype=torch.float32).cuda(0 if torch.cuda.device_count() == 1 else 1)
        larray = np.random.rand(1024, 1024).astype(np.float32)
        shm_buf_torch.fill(ltensor)
        shm_buf_np.fill(larray)

        res = timeit.repeat(lambda shm_buf_torch=shm_buf_torch: shm_buf_torch.fill(ltensor), repeat=5, number=1000)
        print("CUDA-shared-tensor (torch) Fill: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))
        res = timeit.repeat(lambda shm_buf_np=shm_buf_np: shm_buf_np.fill(larray), repeat=5, number=1000)
        print("CUDA-shared-tensor (numpy) Fill: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))

        rtensor = shm_buf_torch.get()
        assert isinstance(rtensor, torch.Tensor)
        assert rtensor.device == torch.device('cuda:0')
        assert rtensor.shape == ltensor.shape
        assert rtensor.dtype == ltensor.dtype

        rarray = shm_buf_np.get()
        assert isinstance(rarray, np.ndarray)
        assert larray.shape == rarray.shape
        assert larray.dtype == rarray.dtype

        event_run.set()

        # Keep producer process running until all consumers exits.
        proc.join()

        del shm_buf_np
        del shm_buf_torch
