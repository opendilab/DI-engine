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

    router = Parallel()
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


class AutoRecover:

    @classmethod
    def main_p0(cls):
        # Wait for p1's message and recovered message from p1
        greets = []
        router = Parallel()
        router.on("greeting_0", lambda msg: greets.append(msg))
        for _ in range(50):
            if greets and greets[-1] == "recovered_p1":
                break
            time.sleep(0.1)
        assert greets and greets[-1] == "recovered_p1"

    @classmethod
    def main_p1(cls):
        # Send empty message to p0
        # When recovered from exception, send recovered_p1 to p0
        # Listen msgs from p2
        greets = []
        router = Parallel()
        router.on("greeting_1", lambda msg: greets.append(msg))

        # Test sending message to p0
        if router._retries == 0:
            for _ in range(10):
                router.emit("greeting_0", "")
                time.sleep(0.1)
            raise Exception("P1 Error")
        elif router._retries == 1:
            for _ in range(10):
                router.emit("greeting_0", "recovered_p1")
                time.sleep(0.1)
        else:
            raise Exception("Failed too many times")

        # Test recover and receving message from p2
        for _ in range(20):
            if greets:
                break
            time.sleep(0.1)
        assert len(greets) > 0

    @classmethod
    def main_p2(cls):
        # Simply send message to p1
        router = Parallel()
        for _ in range(20):
            router.emit("greeting_1", "")
            time.sleep(0.1)

    @classmethod
    def main(cls):
        router = Parallel()
        if router.node_id == 0:
            cls.main_p0()
        elif router.node_id == 1:
            cls.main_p1()
        elif router.node_id == 2:
            cls.main_p2()
        else:
            raise Exception("Invalid node id")


@pytest.mark.unittest
def test_auto_recover():
    # With max_retries=1
    Parallel.runner(n_parallel_workers=3, topology="mesh", auto_recover=True, max_retries=1)(AutoRecover.main)
    # With max_retries=0
    with pytest.raises(Exception) as exc_info:
        Parallel.runner(n_parallel_workers=3, topology="mesh", auto_recover=True, max_retries=0)(AutoRecover.main)
    e = exc_info._excinfo[1]
    assert "P1 Error" in str(e)
