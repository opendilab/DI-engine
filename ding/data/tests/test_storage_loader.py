import os
import pytest
import tempfile
import shutil
import numpy as np
from ding.data.storage_loader import FileStorageLoader
from time import sleep, time
from os import path


@pytest.mark.unittest
def test_file_storage_loader():
    tempdir = path.join(tempfile.gettempdir(), "test_storage_loader")
    loader = FileStorageLoader(dirname=tempdir)
    try:
        total_num = 200
        storages = []
        for i in range(10):
            # 21MB
            data = [
                {
                    "s": "abc",
                    "obs": np.random.rand(4, 84, 84).astype(np.float32),
                    "next_obs": np.random.rand(4, 84, 84).astype(np.float32)
                } for _ in range(96)
            ]
            storage = loader.to_storage(data)
            storages.append(storage)

        start = time()
        for i in range(total_num):
            storage = storages[i % 10]
            data = storage.load()
        origin_time_cost = time() - start
        print("Load time cost: {:.4f}s".format(origin_time_cost))

        call_times = 0

        def callback(data):
            assert data[0]['obs'] is not None
            nonlocal call_times
            call_times += 1

        # First initialize shared memory is very slow, discard this time cost.
        start = time()
        loader._first_meet(storage=storages[0], callback=callback)
        print("Initialize shared memory time: {:.4f}s".format(time() - start))

        start = time()
        for i in range(1, total_num):
            storage = storages[i % 10]
            loader.load(storage, callback)

        while True:
            if call_times == total_num:
                break
            sleep(0.01)
        new_time_cost = time() - start
        print("Loader time cost: {:.4f}s".format(new_time_cost))

        assert new_time_cost < origin_time_cost
    finally:
        print(tempdir)
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
        loader.shutdown()


@pytest.mark.unittest
def test_file_storage_loader_cleanup():
    tempdir = path.join(tempfile.gettempdir(), "test_storage_loader")
    loader = FileStorageLoader(dirname=tempdir, ttl=1)
    try:
        storages = []
        for _ in range(4):
            data = np.random.rand(4, 84, 84).astype(np.float32)
            storage = loader.to_storage(data)
            storages.append(storage)
            sleep(0.5)
        assert len(os.listdir(tempdir)) < 4
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
        loader.shutdown()
