import pytest
from typing import Callable
from ding.worker.buffer import Buffer
from ding.worker.buffer import MemoryStorage


class RateLimit:
    r"""
    Add rate limit threshold to push function
    """

    def __init__(self, max_rate: int = float("inf"), window_seconds: int = 30) -> None:
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.buffered = []

    def handler(self) -> Callable:

        def _handler(buffer: Buffer, action: str, *args):
            if action == "push":
                return self.push(*args)
            return args

        return _handler

    def push(self, data) -> None:
        import time
        current = time.time()
        # Cut off stale records
        self.buffered = [t for t in self.buffered if t > current - self.window_seconds]
        if len(self.buffered) < self.max_rate:
            self.buffered.append(current)
            return False, data
        else:
            return True, None


@pytest.mark.unittest
def test_naive_push_sample():
    # Push and sample
    storage = MemoryStorage(maxlen=10)
    buffer = Buffer(storage)
    for i in range(20):
        buffer.push(i)
    assert storage.count() == 10
    assert len(set(buffer.sample(10))) == 10
    assert 0 not in buffer.sample(10)

    # Clear
    buffer.clear()
    assert storage.count() == 0


@pytest.mark.unittest
def test_rate_limit_push_sample():
    storage = MemoryStorage(maxlen=10)
    ratelimit = RateLimit(max_rate=5)
    buffer = Buffer(storage).use(ratelimit.handler())
    for i in range(10):
        buffer.push(i)
    assert storage.count() == 5
    assert 5 not in buffer.sample(5)


@pytest.mark.unittest
def test_buffer_view():
    storage = MemoryStorage(maxlen=10)
    buf1 = Buffer(storage)
    for i in range(1):
        buf1.push(i)
    assert storage.count() == 1

    ratelimit = RateLimit(max_rate=5)
    buf2 = buf1.view().use(ratelimit.handler())

    for i in range(10):
        buf2.push(i)
    # With 1 record written by buf1 and 5 records written by buf2
    assert len(buf1.middlewares) == 0
    assert storage.count() == 6
