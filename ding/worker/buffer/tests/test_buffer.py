import pytest
from ding.worker.buffer import NaiveBuffer
from ding.worker.buffer import MemoryStorage


@pytest.mark.unittest
def test_naive_push_sample():
    storage = MemoryStorage(maxlen=10)
    naive_buffer = NaiveBuffer(storage)
    for i in range(20):
        naive_buffer.push(i)
    assert storage.count() == 10
    assert len(set(naive_buffer.sample(10))) == 10
    assert 0 not in naive_buffer.sample(10)
