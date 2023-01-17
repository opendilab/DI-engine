import multiprocessing as mp
import pytest
import socket
import torch
from ding.framework.message_queue.perfs.perf_nng import nng_perf_main


@pytest.mark.benchmark
@pytest.mark.multiprocesstest
@pytest.mark.cudatest
def test_nng():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        address = socket.gethostbyname(socket.gethostname())
        params = [
            ("12960", None, address, "learner", "0"),
            ("12961", "tcp://{}:12960".format(address), "127.0.0.1", "collector", "1")
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            pool.starmap(nng_perf_main, params)
            pool.close()
            pool.join()
