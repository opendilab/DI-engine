import pytest
import torch
from ding.worker.buffer import Buffer, MemoryStorage
from ding.worker.buffer.middlewares import clone_object


@pytest.mark.unittest
def test_clone_object():
    buffer = Buffer(MemoryStorage(maxlen=10)).use(clone_object())

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
