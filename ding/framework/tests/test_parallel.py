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
        router.on("test_callback", test_callback)
        # Wait for nodes to bind
        time.sleep(0.7)
        for _ in range(30):
            router.emit("test_callback", "ping")
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


def uncaught_exception_main():
    router = Parallel()
    if router.node_id == 0:
        time.sleep(0.1)
        raise Exception("uncaught exception")
    else:
        time.sleep(0.2)


@pytest.mark.unittest
def test_uncaught_exception():
    # Make one process crash, then the parent process will also crash and output the stack of the wrong process.
    with pytest.raises(Exception) as exc_info:
        Parallel.runner(n_parallel_workers=2, topology="mesh")(uncaught_exception_main)
    e = exc_info._excinfo[1]
    assert "uncaught exception" in str(e)


def disconnected_main():
    router = Parallel()

    if router.node_id == 0:
        # Receive two messages then exit
        greets = []
        router.on("greeting", lambda: greets.append("."))
        for _ in range(10):
            if len(greets) == 1:
                break
            else:
                time.sleep(0.1)
        assert len(greets) > 0
    else:
        # Send 10 greetings even if the target process is exited
        for i in range(10):
            router.emit("greeting")
            time.sleep(0.1)
        assert i == 9


@pytest.mark.unittest
def test_disconnected():
    # Make one process exit normally and the rest will still run, even if the network request
    # is not received by other processes.
    Parallel.runner(n_parallel_workers=2, topology="mesh")(disconnected_main)
