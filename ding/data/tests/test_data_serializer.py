import os
import pickle
import pytest
import torch
import tempfile
import shutil
import numpy as np
from ding.data.data_serializer import DataSerializer, StorageLoader
from ding.data.storage.file import FileStorage
from time import sleep, time
from typing import List
from os import path


@pytest.mark.unittest
def test_big_data():
    # 40MB
    data = [{"s": "abc", "obs": np.random.rand(1024, 1024).astype(np.float32)} for _ in range(10)]
    tempdir = path.join(tempfile.gettempdir(), "test_data_serializer")
    try:
        if not path.exists(tempdir):
            os.mkdir(tempdir)
        data_serializer = DataSerializer(dirname=tempdir).start()

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
        assert isinstance(dump_data, bytes)
        assert isinstance(pickle.loads(dump_data), FileStorage)

        start = time()
        data_serializer.load(dump_data, load_callback)
        assert time() - start < 1
        print("Load time big {:.4f}".format(time() - start))
        sleep(1)
        assert isinstance(load_data, List)

        data_serializer.stop()
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_small_data():
    data = [{"s": "abc", "obs": np.random.rand(10, 10).astype(np.float32)} for _ in range(10)]
    tempdir = path.join(tempfile.gettempdir(), "test_data_serializer")
    try:
        if not path.exists(tempdir):
            os.mkdir(tempdir)
        data_serializer = DataSerializer(dirname=tempdir).start()

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
        assert isinstance(dump_data, bytes)
        assert isinstance(pickle.loads(dump_data), List)

        start = time()
        data_serializer.load(dump_data, load_callback)
        assert time() - start < 1
        print("Load time small {:.4f}".format(time() - start))

        sleep(1)
        assert isinstance(load_data, List)

        data_serializer.stop()
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_storage_loader():
    loader = StorageLoader()
    tempdir = path.join(tempfile.gettempdir(), "test_data_serializer")

    try:
        if not path.exists(tempdir):
            os.mkdir(tempdir)

        total_num = 200
        storages = []
        for i in range(10):
            storage = FileStorage(path.join(tempdir, "data_{}.pkl".format(i)))
            # 21MB
            data = [
                {
                    "s": "abc",
                    "obs": np.random.rand(4, 84, 84).astype(np.float32),
                    "next_obs": np.random.rand(4, 84, 84).astype(np.float32)
                } for _ in range(96)
            ]
            storage.save(data)
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
