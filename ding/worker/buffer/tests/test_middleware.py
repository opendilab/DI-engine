import pytest
import torch
from ding.worker.buffer import Buffer, DequeStorage
from ding.worker.buffer.middleware import clone_object, use_time_check


@pytest.mark.unittest
def test_clone_object():
    buffer = Buffer(DequeStorage(maxlen=10)).use(clone_object())

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


@pytest.mark.tmp
def test_use_time_check():

    def get_data():
        return {'obs': torch.randn(4), 'reward': torch.randn(1), 'info': 'xxx'}

    N = 6
    buffer = Buffer(DequeStorage(maxlen=10)).use(use_time_check(max_use=2))

    for _ in range(N):
        buffer.push(get_data())

    for i in range(2):
        data = buffer.sample(size=N, replace=False)
        assert len(data) == N
        print('sample i')
    buffer.sample(size=6, replace=False)
