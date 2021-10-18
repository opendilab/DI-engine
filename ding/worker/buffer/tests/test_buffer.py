import pytest
from ding.worker.buffer import Buffer
from ding.worker.buffer import MemoryStorage
from ding.worker.buffer.buffer import RateLimit


@pytest.mark.unittest
def test_naive_push_sample():
    storage = MemoryStorage(maxlen=10)
    buffer = Buffer(storage)
    for i in range(20):
        buffer.push(i)
    assert storage.count() == 10
    assert len(set(buffer.sample(10))) == 10
    assert 0 not in buffer.sample(10)


@pytest.mark.unittest
def test_rate_limit_push_sample():
    storage = MemoryStorage(maxlen=10)
    ratelimit = RateLimit(max_rate=5)
    buffer = Buffer(storage).use(ratelimit.handler())
    for i in range(10):
        buffer.push(i)
    assert storage.count() == 5
    assert 5 not in buffer.sample(5)
