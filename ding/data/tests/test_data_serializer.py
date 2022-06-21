from time import sleep, time
import pytest
import torch
import multiprocessing as mp
from ding.data.data_serializer import DataSerializer

# 20MB
data = [{"s": "abc", "obs": torch.rand(1024, 1024)} for _ in range(20)]


@pytest.mark.unittest
def test_data_serializer():
    data_serializer = DataSerializer().start()

    dump_data = None
    load_data = None

    def dump_callback(s):
        nonlocal dump_data
        dump_data = s
        sleep(1)

    def load_callback(obj):
        nonlocal load_data
        load_data = obj
        sleep(1)

    # Test non-block dump and load
    start = time()
    data_serializer.dump(data, dump_callback)
    assert time() - start < 1
    sleep(1)
    assert dump_data is not None

    start = time()
    data_serializer.load(dump_data, load_callback)
    assert time() - start < 1
    sleep(1)
    assert load_data is not None

    data_serializer.stop()
