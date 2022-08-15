from ding.framework.message_queue.perfs.perf_torchrpc_nccl import rpc_model_exchanger
from ding.compatibility import torch_ge_1121
import multiprocessing as mp
import pytest
import torch
import platform


@pytest.mark.unittest
@pytest.mark.benchmark
@pytest.mark.cudatest
def test_shm_numpy_shm():
    if platform.system().lower() != 'windows' and torch.cuda.is_available():
        if torch_ge_1121() and torch.cuda.device_count() >= 2:
            params = [(0, "tcp://127.0.0.1:12345", False, True), (1, "tcp://127.0.0.1:12345", False, True)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=2) as pool:
                pool.starmap(rpc_model_exchanger, params)
