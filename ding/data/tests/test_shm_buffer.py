import pytest
import numpy as np
import timeit
import torch
import time
from ding.data.shm_buffer import ShmBuffer, ShmBufferCuda


def subprocess_np_shm(shm_buf):
    data = np.random.rand(1024, 1024).astype(np.float32)
    res = timeit.repeat(lambda: shm_buf.fill(data), repeat=5, number=1000)
    print("Mean: {:.4f}s, STD: {:.4f}s, Mean each call: {:.4f}ms".format(np.mean(res), np.std(res), np.mean(res)))


def subprocess_cuda_shared_tensor(shm_buf_np, shm_buf_torch, event_wait, event_fire, copy_on_get):
    event_wait.wait()
    event_wait.clear()
    rtensor = shm_buf_torch.get()
    assert isinstance(rtensor, torch.Tensor)
    assert rtensor.device == torch.device('cuda:0')
    assert rtensor.dtype == torch.float32
    assert rtensor.sum().item() == 1024 * 1024

    rarray = shm_buf_np.get()
    assert isinstance(rarray, np.ndarray)
    assert rarray.dtype == np.dtype(np.float32)
    assert rarray.dtype == np.dtype(np.float32)
    assert rtensor.sum() == 1024 * 1024

    shm_buf_torch.fill(torch.zeros((1024, 1024), dtype=torch.float32, device=torch.device('cuda:0')))
    shm_buf_np.fill(np.zeros((1024, 1024), dtype=np.float32))

    event_fire.set()

    if copy_on_get:
        event_wait.wait()
        shm_buf_torch.buffer[0] = 9.0
        shm_buf_np.buffer[0] = 9.0
        event_fire.set()

    del shm_buf_np
    del shm_buf_torch


def subprocess_cuda_shared_tensor_case2(shm_buf_np, shm_buf_torch, event_wait):
    event_wait.wait()
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
@pytest.mark.cudatest
# @pytest.mark.multiprocesstest
@pytest.mark.parametrize("copy_on_get", [True, False])
def test_cuda_shm(copy_on_get):
    if torch.cuda.is_available():
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')

        event_fire, event_wait = ctx.Event(), ctx.Event()
        shm_buf_np = ShmBufferCuda(np.dtype(np.float32), shape=(1024, 1024), copy_on_get=copy_on_get)
        shm_buf_torch = ShmBufferCuda(torch.float32, shape=(1024, 1024), copy_on_get=copy_on_get)
        proc = ctx.Process(
            target=subprocess_cuda_shared_tensor, args=[shm_buf_np, shm_buf_torch, event_fire, event_wait, copy_on_get]
        )
        proc.start()

        ltensor = torch.ones((1024, 1024), dtype=torch.float32, device=torch.device('cuda:0'))
        larray = np.ones((1024, 1024), dtype=np.float32)
        shm_buf_torch.fill(ltensor)
        shm_buf_np.fill(larray)

        rtensor = shm_buf_torch.get()
        assert isinstance(rtensor, torch.Tensor)
        assert rtensor.device == torch.device('cuda:0')
        assert rtensor.shape == ltensor.shape
        assert rtensor.dtype == ltensor.dtype
        assert rtensor.sum().item() == 1024 * 1024

        rarray = shm_buf_np.get()
        assert isinstance(rarray, np.ndarray)
        assert larray.shape == rarray.shape
        assert larray.dtype == rarray.dtype
        assert larray.sum() == 1024 * 1024

        event_fire.set()
        event_wait.wait()
        event_wait.clear()
        rtensor = shm_buf_torch.get()
        assert isinstance(rtensor, torch.Tensor)
        assert rtensor.device == torch.device('cuda:0')
        assert rtensor.shape == ltensor.shape
        assert rtensor.dtype == ltensor.dtype
        assert rtensor.sum().item() == 0

        rarray = shm_buf_np.get()
        assert isinstance(rarray, np.ndarray)
        assert rarray.shape == larray.shape
        assert rarray.dtype == larray.dtype
        assert rarray.sum() == 0

        if copy_on_get:
            event_fire.set()
            event_wait.wait()
            assert shm_buf_torch.buffer[0].item() == 9.0
            assert shm_buf_np.buffer[0] == 9.0

        # Keep producer process running until all consumers exits.
        proc.join()

        del shm_buf_np
        del shm_buf_torch


@pytest.mark.benchmark
@pytest.mark.cudatest
# @pytest.mark.multiprocesstest
@pytest.mark.parametrize("copy_on_get", [True, False])
def test_cudabuff_perf(copy_on_get):
    if torch.cuda.is_available():
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')

        event_fire, event_wait = ctx.Event(), ctx.Event()
        shm_buf_np = ShmBufferCuda(np.dtype(np.float32), shape=(1024, 1024), copy_on_get=copy_on_get)
        shm_buf_torch = ShmBufferCuda(torch.float32, shape=(1024, 1024), copy_on_get=copy_on_get)
        proc = ctx.Process(target=subprocess_cuda_shared_tensor_case2, args=[shm_buf_np, shm_buf_torch, event_fire])
        proc.start()

        ltensor = torch.ones((1024, 1024), dtype=torch.float32, device=torch.device('cuda:0'))
        larray = np.ones((1024, 1024), dtype=np.float32)
        shm_buf_torch.fill(ltensor)
        shm_buf_np.fill(larray)

        res = timeit.repeat(lambda shm_buf_torch=shm_buf_torch: shm_buf_torch.fill(ltensor), repeat=5, number=1000)
        print("CUDA-shared-tensor (torch) Fill: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))
        res = timeit.repeat(lambda shm_buf_np=shm_buf_np: shm_buf_np.fill(larray), repeat=5, number=1000)
        print("CUDA-shared-tensor (numpy) Fill: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))

        res = timeit.repeat(lambda shm_buf_torch=shm_buf_torch: shm_buf_torch.get(), repeat=5, number=1000)
        print("CUDA-shared-tensor (torch) Get: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))
        res = timeit.repeat(lambda shm_buf_np=shm_buf_np: shm_buf_np.get(), repeat=5, number=1000)
        print("CUDA-shared-tensor (numpy) Get: mean: {:.4f}s, STD: {:.4f}s".format(np.mean(res), np.std(res)))
        event_fire.set()
        proc.join()

        del shm_buf_np
        del shm_buf_torch
