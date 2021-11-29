from collections import defaultdict
import pytest
import time
import asyncio
from ding.framework import Parallel


def parallel_main():
    msg = defaultdict(bool)
    router = Parallel()

    def test_callback(key):
        msg[key] = True

    router.register_rpc("test_callback", test_callback)
    # Wait for rpc function to bind (send to thread)
    time.sleep(0.1)

    router.send_rpc("test_callback", "ping")

    for _ in range(30):
        if msg["ping"]:
            break
        time.sleep(0.03)
    assert msg["ping"]

    asyncio.run(router.asend_rpc("test_callback", "pong"))
    for _ in range(30):
        if msg["pong"]:
            break
        time.sleep(0.03)
    assert msg["pong"]


@pytest.mark.unittest
def test_parallel_run():
    Parallel.runner(n_parallel_workers=2)(parallel_main)
    Parallel.runner(n_parallel_workers=2, protocol="tcp")(parallel_main)
