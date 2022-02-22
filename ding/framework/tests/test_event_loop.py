from time import sleep
import pytest
from ding.framework import EventLoop
from threading import Lock


@pytest.mark.unittest
def test_event_loop():
    loop = EventLoop.get_event_loop("test")
    try:
        counter = 0
        lock = Lock()

        def callback(n, lock):
            nonlocal counter
            with lock:
                counter += n

        # Test on
        loop.on("count", callback)

        for i in range(5):
            loop.emit("count", i, lock)
        sleep(0.1)
        assert counter == 10

        # Test off
        loop.off("count")
        loop.emit("count", 10, lock)
        sleep(0.1)
        assert counter == 10

        # Test once
        loop.once("count", callback)
        loop.emit("count", 10, lock)
        sleep(0.1)
        assert counter == 20
        loop.emit("count", 10, lock)
        assert counter == 20

        # Test exception
        def except_callback():
            raise Exception("error")

        loop.on("error", except_callback)
        loop.emit("error")
        sleep(0.1)
        assert loop._exception is not None
        with pytest.raises(Exception):
            loop.emit("error")
    finally:
        loop.stop()
