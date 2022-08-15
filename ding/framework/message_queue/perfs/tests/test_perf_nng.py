from ding.framework.message_queue.perfs.perf_nng import nng_perf_main
import multiprocessing as mp
import pytest


@pytest.mark.unittest
@pytest.mark.benchmark
def test_nng():
    params = [
        ("12345", None, "127.0.0.1", "learner", "0"), ("12346", "tcp://127.0.0.1:12345", "127.0.0.1", "collector", "1")
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        pool.starmap(nng_perf_main, params)
