import os
import pickle
import pytest
import torch
import tempfile
import shutil
from ding.data import DataSerializer, FileStorage
from time import sleep, time
from typing import List
from os import path


@pytest.mark.unittest
def test_big_data():
    # 40MB
    data = [{"s": "abc", "obs": torch.rand(1024, 1024)} for _ in range(10)]
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
        sleep(1)
        assert isinstance(load_data, List)

        data_serializer.stop()
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_small_data():
    data = [{"s": "abc", "obs": torch.rand(10, 10)} for _ in range(10)]
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
        sleep(1)
        assert isinstance(load_data, List)

        data_serializer.stop()
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
