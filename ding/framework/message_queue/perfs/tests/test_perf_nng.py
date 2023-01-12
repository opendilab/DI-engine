from ding.framework.message_queue.perfs.perf_nng import nng_perf_main
import multiprocessing as mp
import pytest


@pytest.mark.benchmark
# @pytest.mark.multiprocesstest
def test_nng():
    params = [
        ("12376", None, "127.0.0.1", "learner", "0"), ("12378", "tcp://127.0.0.1:12376", "127.0.0.1", "collector", "1")
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        pool.starmap(nng_perf_main, params)
