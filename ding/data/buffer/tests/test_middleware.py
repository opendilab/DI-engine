import pytest
import torch
from ding.data.buffer import DequeBuffer
from ding.data.buffer.middleware import clone_object, use_time_check, staleness_check, sample_range_view
from ding.data.buffer.middleware import PriorityExperienceReplay, group_sample
from ding.data.buffer.middleware.padding import padding


@pytest.mark.unittest
def test_clone_object():
    buffer = DequeBuffer(size=10).use(clone_object())

    # Store a dict, a list, a tensor
    arr = [{"key": "v1"}, ["a"], torch.Tensor([1, 2, 3])]
    for o in arr:
        buffer.push(o)

    # Modify it
    for item in buffer.sample(len(arr)):
        item = item.data
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
        item = item.data
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

    for _ in range(2):
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
    buffer.use(PriorityExperienceReplay(buffer, IS_weight=True))
    for _ in range(N):
        buffer.push(get_data(), meta={'priority': 2.0})
    assert buffer.count() == N
    for _ in range(N):
        buffer.push(get_data(), meta={'priority': 2.0})
    assert buffer.count() == N + N
    data = buffer.sample(size=N + N, replace=False)
    assert len(data) == N + N
    for item in data:
        meta = item.meta
        assert set(meta.keys()).issuperset(set(['priority', 'priority_idx', 'priority_IS']))
        meta['priority'] = 3.0
    for item in data:
        data, index, meta = item.data, item.index, item.meta
        buffer.update(index, data, meta)
    data = buffer.sample(size=1)
    assert data[0].meta['priority'] == 3.0
    buffer.delete(data[0].index)
    assert buffer.count() == N + N - 1
    buffer.clear()
    assert buffer.count() == 0


@pytest.mark.unittest
def test_padding():
    buffer = DequeBuffer(size=10)
    buffer.use(padding())
    for i in range(10):
        buffer.push(i, {"group": i & 5})  # [3,3,2,2]
    sampled_data = buffer.sample(4, groupby="group")
    assert len(sampled_data) == 4
    for grouped_data in sampled_data:
        assert len(grouped_data) == 3


@pytest.mark.unittest
def test_group_sample():
    buffer = DequeBuffer(size=10)
    buffer.use(padding(policy="none")).use(group_sample(size_in_group=5, ordered_in_group=True, max_use_in_group=True))
    for i in range(4):
        buffer.push(i, {"episode": 0})
    for i in range(6):
        buffer.push(i, {"episode": 1})
    sampled_data = buffer.sample(2, groupby="episode")
    assert len(sampled_data) == 2

    def check_group0(grouped_data):
        # In group0 should find only last record with data as None
        n_none = 0
        for item in grouped_data:
            if item.data is None:
                n_none += 1
        assert n_none == 1

    def check_group1(grouped_data):
        # In group1 every record should have data and meta
        for item in grouped_data:
            assert item.data is not None

    for grouped_data in sampled_data:
        assert len(grouped_data) == 5
        meta = grouped_data[0].meta
        if meta and "episode" in meta and meta["episode"] == 1:
            check_group1(grouped_data)
        else:
            check_group0(grouped_data)


@pytest.mark.unittest
def test_sample_range_view():
    buffer_ = DequeBuffer(size=10)
    for i in range(5):
        buffer_.push({'data': 'x'})
    for i in range(5, 5 + 3):
        buffer_.push({'data': 'y'})
    for i in range(8, 8 + 2):
        buffer_.push({'data': 'z'})

    buffer1 = buffer_.view()
    buffer1.use(sample_range_view(buffer1, start=-5, end=-2))
    for _ in range(10):
        sampled_data = buffer1.sample(1)
        assert sampled_data[0].data['data'] == 'y'

    buffer2 = buffer_.view()
    buffer2.use(sample_range_view(buffer1, start=-2))
    for _ in range(10):
        sampled_data = buffer2.sample(1)
        assert sampled_data[0].data['data'] == 'z'
