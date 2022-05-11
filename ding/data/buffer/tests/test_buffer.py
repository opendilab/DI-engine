import pytest
import time
import random
import functools
from typing import Callable
from ding.data.buffer import DequeBuffer
from ding.data.buffer.buffer import BufferedData
from torch.utils.data import DataLoader


class RateLimit:
    r"""
    Add rate limit threshold to push function
    """

    def __init__(self, max_rate: int = float("inf"), window_seconds: int = 30) -> None:
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.buffered = []

    def __call__(self, action: str, chain: Callable, *args, **kwargs):
        if action == "push":
            return self.push(chain, *args, **kwargs)
        return chain(*args, **kwargs)

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
    assert len(buffer.sample(5, sample_range=slice(5, 10))) == 5
    assert 0 not in [item.data for item in buffer.sample(5, sample_range=slice(5, 10))]


@pytest.mark.unittest
def test_rate_limit_push_sample():
    buffer = DequeBuffer(size=10).use(RateLimit(max_rate=5))
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

    buf2 = buf1.view().use(RateLimit(max_rate=5)).use(add_10())

    for i in range(10):
        buf2.push(i)
    # With 1 record written by buf1 and 5 records written by buf2
    assert len(buf1._middleware) == 0
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
def test_update():
    buf = DequeBuffer(size=10)
    for i in range(1):
        buf.push({"data": i}, {"meta": i})

    # Update one data
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

    # When exceed buffer size
    for i in range(20):
        buf.push({"data": i})
    assert len(buf.indices) == 10
    assert len(buf.storage) == 10
    for i in range(10):
        index = buf.storage[i].index
        assert buf.indices.get(index) == i


@pytest.mark.unittest
def test_delete():
    maxlen = 100
    cumlen = 40
    dellen = 20
    buf = DequeBuffer(size=maxlen)
    for i in range(cumlen):
        buf.push(i)
    # Delete data
    del_indices = [item.index for item in buf.sample(dellen)]
    buf.delete(del_indices)
    # Reappend
    for i in range(10):
        buf.push(i)
    remlen = min(cumlen, maxlen) - dellen + 10
    assert len(buf.indices) == remlen
    assert len(buf.storage) == remlen
    for i in range(remlen):
        index = buf.storage[i].index
        assert buf.indices.get(index) == i


@pytest.mark.unittest
def test_ignore_insufficient():
    buffer = DequeBuffer(size=10)
    for i in range(2):
        buffer.push(i)

    with pytest.raises(ValueError):
        buffer.sample(3, ignore_insufficient=False)
    data = buffer.sample(3, ignore_insufficient=True)
    assert len(data) == 0


@pytest.mark.unittest
def test_independence():
    # By replace
    buffer = DequeBuffer(size=1)
    data = {"key": "origin"}
    buffer.push(data)
    sampled_data = buffer.sample(2, replace=True)
    assert len(sampled_data) == 2
    sampled_data[0].data["key"] = "new"
    assert sampled_data[1].data["key"] == "origin"

    # By indices
    buffer = DequeBuffer(size=1)
    data = {"key": "origin"}
    buffered = buffer.push(data)
    indices = [buffered.index, buffered.index]
    sampled_data = buffer.sample(indices=indices)
    assert len(sampled_data) == 2
    sampled_data[0].data["key"] = "new"
    assert sampled_data[1].data["key"] == "origin"


@pytest.mark.unittest
def test_groupby():
    buffer = DequeBuffer(size=3)
    buffer.push("a", {"group": 1})
    buffer.push("b", {"group": 2})
    buffer.push("c", {"group": 2})

    sampled_data = buffer.sample(2, groupby="group")
    assert len(sampled_data) == 2
    group1 = sampled_data[0] if len(sampled_data[0]) == 1 else sampled_data[1]
    group2 = sampled_data[0] if len(sampled_data[0]) == 2 else sampled_data[1]
    # Group1 should contain a
    assert "a" == group1[0].data
    # Group2 should contain b and c
    data = [buffered.data for buffered in group2]  # ["b", "c"]
    assert "b" in data
    assert "c" in data

    # Push new data and swap out a, the result will all in group 2
    buffer.push("d", {"group": 2})
    sampled_data = buffer.sample(1, groupby="group")
    assert len(sampled_data) == 1
    assert len(sampled_data[0]) == 3
    data = [buffered.data for buffered in sampled_data[0]]
    assert "d" in data

    # Update meta, set first data's group to 1
    first: BufferedData = buffer.storage[0]
    buffer.update(first.index, first.data, {"group": 1})
    sampled_data = buffer.sample(2, groupby="group")
    assert len(sampled_data) == 2

    # Delete last record, each group will only have one record
    last: BufferedData = buffer.storage[-1]
    buffer.delete(last.index)
    sampled_data = buffer.sample(2, groupby="group")
    assert len(sampled_data) == 2


@pytest.mark.unittest
def test_import_export():
    buffer = DequeBuffer(size=10)
    data_with_meta = [(i, {}) for i in range(10)]
    buffer.import_data(data_with_meta)
    assert buffer.count() == 10

    sampled_data = buffer.export_data()
    assert len(sampled_data) == 10


@pytest.mark.unittest
def test_dataset():
    buffer = DequeBuffer(size=10)
    for i in range(10):
        buffer.push(i)
    dataloader = DataLoader(buffer, batch_size=6, shuffle=True, collate_fn=lambda batch: batch)
    for batch in dataloader:
        assert len(batch) in [4, 6]


@pytest.mark.unittest
def test_unroll_len_in_group():
    buffer = DequeBuffer(size=100)
    for i in range(10):
        for env_id in list("ABC"):
            buffer.push(i, {"env": env_id})

    sampled_data = buffer.sample(3, groupby="env", unroll_len=4)
    assert len(sampled_data) == 3
    for grouped_data in sampled_data:
        assert len(grouped_data) == 4
        # Ensure each group has the same env
        env_ids = set(map(lambda sample: sample.meta["env"], grouped_data))
        assert len(env_ids) == 1
        # Ensure samples in each group is continuous
        result = functools.reduce(lambda a, b: a and a.data + 1 == b.data and b, grouped_data)
        assert isinstance(result, BufferedData), "Not continuous"


@pytest.mark.unittest
def test_insufficient_unroll_len_in_group():
    buffer = DequeBuffer(size=100)

    num = 3  # Items in group A,B,C is 3,4,5
    for env_id in list("ABC"):
        for i in range(num):
            buffer.push(i, {"env": env_id})
        num += 1

    with pytest.raises(ValueError) as exc_info:
        buffer.sample(3, groupby="env", unroll_len=4)
    e = exc_info._excinfo[1]
    assert "There are less than" in str(e)

    # Sample with replace
    sampled_data = buffer.sample(3, groupby="env", unroll_len=4, replace=True)
    assert len(sampled_data) == 3
    for grouped_data in sampled_data:
        assert len(grouped_data) == 4
        # Ensure each group has the same env
        env_ids = set(map(lambda sample: sample.meta["env"], grouped_data))
        assert len(env_ids) == 1
        # Ensure samples in each group is continuous
        result = functools.reduce(lambda a, b: a and a.data + 1 == b.data and b, grouped_data)
        assert isinstance(result, BufferedData), "Not continuous"
