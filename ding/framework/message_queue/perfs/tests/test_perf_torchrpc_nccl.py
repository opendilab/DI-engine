from ding.framework.message_queue.perfs.perf_torchrpc_nccl import rpc_model_exchanger
from ding.compatibility import torch_ge_1121
import multiprocessing as mp
import pytest
import torch
import platform


@pytest.mark.mqbenchmark
@pytest.mark.cudatest
@pytest.mark.multiprocesstest
def test_perf_torchrpc_nccl():
    if platform.system().lower() != 'windows' and torch.cuda.is_available():
        if torch_ge_1121() and torch.cuda.device_count() >= 2:
            params = [(0, "tcp://127.0.0.1:12387", False, True), (1, "tcp://127.0.0.1:12387", False, True)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=2) as pool:
                pool.starmap(rpc_model_exchanger, params)
