import multiprocessing as mp
import pytest
import torch
import platform
import socket
from ding.utils.system_helper import find_free_port
from ding.framework.message_queue.perfs.perf_torchrpc_nccl import rpc_model_exchanger
from ding.compatibility import torch_ge_1121


@pytest.mark.benchmark
@pytest.mark.cudatest
# @pytest.mark.multiprocesstest
def test_perf_torchrpc_nccl():
    address = socket.gethostbyname(socket.gethostname())
    init_method = "tcp://{}:{}".format(address, find_free_port(address))
    if platform.system().lower() != 'windows' and torch.cuda.is_available():
        if torch_ge_1121() and torch.cuda.device_count() >= 2:
            params = [(0, init_method, False, True), (1, init_method, False, True)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=2) as pool:
                pool.starmap(rpc_model_exchanger, params)
                pool.close()
                pool.join()
