from asyncio.tasks import sleep
from collections import defaultdict
import pytest
import time
import asyncio
from ding.framework import Parallel


@pytest.mark.unittest
def test_parallel_run():
    router = Parallel()

    def main():
        msg = defaultdict(bool)

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

    router.run(main, n_workers=2)
    router.run(main, n_workers=2, protocol="tcp")
