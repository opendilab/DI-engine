from collections import defaultdict
import pytest
import time
import os
from ding.framework import Parallel
from ding.utils.design_helper import SingletonMetaclass


def parallel_main():
    msg = defaultdict(bool)

    def test_callback(key):
        msg[key] = True

    with Parallel() as router:
        router.register_rpc("test_callback", test_callback)
        # Wait for nodes to bind
        time.sleep(0.7)

        router.send_rpc("test_callback", "ping")

        for _ in range(30):
            if msg["ping"]:
                break
            time.sleep(0.03)
        assert msg["ping"]
        # Avoid can not receiving messages from each other after exit parallel
        time.sleep(0.7)


@pytest.mark.unittest
def test_parallel_run():
    Parallel.runner(n_parallel_workers=2)(parallel_main)
    Parallel.runner(n_parallel_workers=2, protocol="tcp")(parallel_main)


def parallel_main_alone(pid):
    assert os.getpid() == pid
    router = Parallel()
    time.sleep(0.3)  # Waiting to bind listening address
    assert router._bind_addr


def test_parallel_run_alone():
    try:
        Parallel.runner(n_parallel_workers=1)(parallel_main_alone, os.getpid())
    finally:
        del SingletonMetaclass.instances[Parallel]


def star_parallel_main():
    with Parallel() as router:
        if router.node_id != 0:
            assert len(router.attach_to) == 1

        # Wait for other nodes
        time.sleep(2)


@pytest.mark.unittest
def test_parallel_topology():
    Parallel.runner(n_parallel_workers=3, topology="star")(star_parallel_main)
