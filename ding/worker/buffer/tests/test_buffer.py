import pytest
import time
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

        def _handler(action: str, next: Callable, *args, **kwargs):
            if action == "push":
                return self.push(next, *args, **kwargs)
            return next(*args, **kwargs)

        return _handler

    def push(self, next, data, *args, **kwargs) -> None:
        current = time.time()
        # Cut off stale records
        self.buffered = [t for t in self.buffered if t > current - self.window_seconds]
        if len(self.buffered) < self.max_rate:
            self.buffered.append(current)
            return next(data, *args, **kwargs)
        else:
            return None


def add_10() -> Callable:
    """
    Transform data on sampling
    """

    def sample(next: Callable, size: int, replace: bool = False, *args, **kwargs):
        print("Replace", replace)
        data = next(size, replace, *args, **kwargs)
        return [d + 10 for d in data]

    def _subview(action: str, next: Callable, *args, **kwargs):
        if action == "sample":
            # print("ARGS", args)
            return sample(next, *args, **kwargs)
        return next(*args, **kwargs)

    return _subview


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

    # Test replace sample
    for i in range(5):
        buffer.push(i)
    assert storage.count() == 5
    assert len(buffer.sample(10, replace=True)) == 10


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
    buf2 = buf1.view().use(ratelimit.handler()).use(add_10())

    for i in range(10):
        buf2.push(i)
    # With 1 record written by buf1 and 5 records written by buf2
    assert len(buf1.middlewares) == 0
    assert storage.count() == 6
    # All data in buffer should bigger than 10 because of `add_10`
    assert all(d >= 10 for d in buf2.sample(5))
    # But data in storage is still less than 10
    assert all(d < 10 for d in storage.sample(5))
