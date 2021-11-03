import pytest
import torch
from ding.worker.buffer import DequeBuffer
from ding.worker.buffer.middleware import clone_object, use_time_check, staleness_check, priority


@pytest.mark.unittest
def test_clone_object():
    buffer = DequeBuffer(size=10).use(clone_object())

    # Store a dict, a list, a tensor
    arr = [{"key": "v1"}, ["a"], torch.Tensor([1, 2, 3])]
    for o in arr:
        buffer.push(o)

    # Modify it
    for item in buffer.sample(len(arr)):
        if isinstance(item, dict):
            item["key"] = "v2"
        elif isinstance(item, list):
            item.append("b")
        elif isinstance(item, torch.Tensor):
            item[0] = 3
        else:
            raise Exception("Unexpected type")

    # Resample it, and check their values
    for item in buffer.sample(len(arr)):
        if isinstance(item, dict):
            assert item["key"] == "v1"
        elif isinstance(item, list):
            assert len(item) == 1
        elif isinstance(item, torch.Tensor):
            assert item[0] == 1
        else:
            raise Exception("Unexpected type")


def get_data():
    return {'obs': torch.randn(4), 'reward': torch.randn(1), 'info': 'xxx'}


@pytest.mark.unittest
def test_use_time_check():
    N = 6
    buffer = DequeBuffer(size=10)
    buffer.use(use_time_check(buffer, max_use=2))

    for _ in range(N):
        buffer.push(get_data())

    for i in range(2):
        data = buffer.sample(size=N, replace=False)
        assert len(data) == N
    with pytest.raises(ValueError):
        buffer.sample(size=1, replace=False)


@pytest.mark.unittest
def test_staleness_check():
    N = 6
    buffer = DequeBuffer(size=10)
    buffer.use(staleness_check(buffer, max_staleness=10))

    with pytest.raises(AssertionError):
        buffer.push(get_data())
    for _ in range(N):
        buffer.push(get_data(), meta={'train_iter_data_collected': 0})
    data = buffer.sample(size=N, replace=False, train_iter_sample_data=9)
    assert len(data) == N
    data = buffer.sample(size=N, replace=False, train_iter_sample_data=10)  # edge case
    assert len(data) == N
    for _ in range(2):
        buffer.push(get_data(), meta={'train_iter_data_collected': 5})
    assert buffer.count() == 8
    with pytest.raises(ValueError):
        data = buffer.sample(size=N, replace=False, train_iter_sample_data=11)
    assert buffer.count() == 2


@pytest.mark.unittest
def test_priority():
    N = 5
    buffer = DequeBuffer(size=10)
    buffer.use(priority(buffer, buffer_size=10, IS_weight=True))
    for _ in range(N):
        buffer.push(get_data())
    assert buffer.count() == N
    for _ in range(N):
        buffer.push(get_data(), meta={'priority': 2.0})
    assert buffer.count() == N + N
    data = buffer.sample(size=N + N, replace=False)
    assert len(data) == N + N
    for (item, _, meta) in data:
        assert set(meta.keys()).issuperset(set(['priority', 'priority_idx', 'priority_IS']))
        meta['priority'] = 3.0
    for item, index, meta in data:
        buffer.update(index, item, meta)
    data = buffer.sample(size=1)
    assert data[0][2]['priority'] == 3.0
    buffer.delete(data[0][1])
    assert buffer.count() == N + N - 1
    buffer.clear()
    assert buffer.count() == 0
