import pytest
import time
import random
from typing import Callable
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.buffer import BufferedData


class RateLimit:
    r"""
    Add rate limit threshold to push function
    """

    def __init__(self, max_rate: int = float("inf"), window_seconds: int = 30) -> None:
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.buffered = []

    def handler(self) -> Callable:

        def _handler(action: str, chain: Callable, *args, **kwargs):
            if action == "push":
                return self.push(chain, *args, **kwargs)
            return chain(*args, **kwargs)

        return _handler

    def push(self, chain, data, *args, **kwargs) -> None:
        current = time.time()
        # Cut off stale records
        self.buffered = [t for t in self.buffered if t > current - self.window_seconds]
        if len(self.buffered) < self.max_rate:
            self.buffered.append(current)
            return chain(data, *args, **kwargs)
        else:
            return None


def add_10() -> Callable:
    """
    Transform data on sampling
    """

    def sample(chain: Callable, size: int, replace: bool = False, *args, **kwargs):
        sampled_data = chain(size, replace, *args, **kwargs)
        return [BufferedData(data=item.data + 10, index=item.index, meta=item.meta) for item in sampled_data]

    def _subview(action: str, chain: Callable, *args, **kwargs):
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _subview


@pytest.mark.unittest
def test_naive_push_sample():
    # Push and sample
    buffer = DequeBuffer(size=10)
    for i in range(20):
        buffer.push(i)
    assert buffer.count() == 10
    assert 0 not in [item.data for item in buffer.sample(10)]

    # Clear
    buffer.clear()
    assert buffer.count() == 0

    # Test replace sample
    for i in range(5):
        buffer.push(i)
    assert buffer.count() == 5
    assert len(buffer.sample(10, replace=True)) == 10

    # Test slicing
    buffer.clear()
    for i in range(10):
        buffer.push(i)
    assert len(buffer.sample(5, range=slice(5, 10))) == 5
    assert 0 not in [item.data for item in buffer.sample(5, range=slice(5, 10))]


@pytest.mark.unittest
def test_rate_limit_push_sample():
    ratelimit = RateLimit(max_rate=5)
    buffer = DequeBuffer(size=10).use(ratelimit.handler())
    for i in range(10):
        buffer.push(i)
    assert buffer.count() == 5
    assert 5 not in buffer.sample(5)


@pytest.mark.unittest
def test_buffer_view():
    buf1 = DequeBuffer(size=10)
    for i in range(1):
        buf1.push(i)
    assert buf1.count() == 1

    ratelimit = RateLimit(max_rate=5)
    buf2 = buf1.view().use(ratelimit.handler()).use(add_10())

    for i in range(10):
        buf2.push(i)
    # With 1 record written by buf1 and 5 records written by buf2
    assert len(buf1.middleware) == 0
    assert buf1.count() == 6
    # All data in buffer should bigger than 10 because of `add_10`
    assert all(d.data >= 10 for d in buf2.sample(5))
    # But data in storage is still less than 10
    assert all(d.data < 10 for d in buf1.sample(5))


@pytest.mark.unittest
def test_sample_with_index():
    buf = DequeBuffer(size=10)
    for i in range(10):
        buf.push({"data": i}, {"meta": i})
    # Random sample and get indices
    indices = [item.index for item in buf.sample(10)]
    assert len(indices) == 10
    random.shuffle(indices)
    indices = indices[:5]

    # Resample by indices
    new_indices = [item.index for item in buf.sample(indices=indices)]
    assert len(new_indices) == len(indices)
    for index in new_indices:
        assert index in indices


@pytest.mark.unittest
def test_update_delete():
    buf = DequeBuffer(size=10)
    for i in range(1):
        buf.push({"data": i}, {"meta": i})

    # Update data
    [item] = buf.sample(1)
    item.data["new_prop"] = "any"
    meta = None
    success = buf.update(item.index, item.data, item.meta)
    assert success
    # Resample
    [item] = buf.sample(1)
    assert "new_prop" in item.data
    assert meta is None
    # Update object that not exists in buffer
    success = buf.update("invalidindex", {}, None)
    assert not success

    # Delete data
    [item] = buf.sample(1)
    buf.delete(item.index)
    assert buf.count() == 0


@pytest.mark.unittest
def test_ignore_insufficient():
    buffer = DequeBuffer(size=10)
    for i in range(2):
        buffer.push(i)

    with pytest.raises(ValueError):
        buffer.sample(3)
    data = buffer.sample(3, ignore_insufficient=True)
    assert len(data) == 0
