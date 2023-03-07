import os
import timeit
import pytest
import tempfile
import shutil
import numpy as np
import torch
import treetensor.torch as ttorch
from ding.data.shm_buffer import ShmBuffer
from ding.data.storage_loader import FileStorageLoader
from time import sleep, time
from os import path
from ding.framework.supervisor import RecvPayload


@pytest.mark.tmp  # gitlab ci and local test pass, github always fail
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
                    # "next_obs": np.random.rand(4, 84, 84).astype(np.float32),
                    # "obs": torch.rand(4, 84, 84, dtype=torch.float32),
                    "next_obs": torch.rand(4, 84, 84, dtype=torch.float32)
                } for _ in range(96)
            ]
            storage = loader.save(data)
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
            storage = loader.save(data)
            storages.append(storage)
            sleep(0.5)
        assert len(os.listdir(tempdir)) < 4
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
        loader.shutdown()


@pytest.mark.unittest
def test_shared_object():
    loader = FileStorageLoader(dirname="")

    # ========== Test array ==========
    obj = [{"obs": np.random.rand(100, 100)} for _ in range(10)]
    shm_obj = loader._create_shm_buffer(obj)
    assert len(shm_obj.buf) == len(obj) * 2
    assert isinstance(shm_obj.buf[0]["obs"], ShmBuffer)

    # Callback
    payload = RecvPayload(proc_id=0, data=obj)
    loader._shm_callback(payload=payload, shm_obj=shm_obj)
    assert len(payload.data) == 10
    assert [d["obs"] is None for d in payload.data]

    # ========== Putback ==========
    loader._shm_putback(payload=payload, shm_obj=shm_obj)
    obj = payload.data
    assert len(obj) == 10
    for o in obj:
        assert isinstance(o["obs"], np.ndarray)
        assert o["obs"].shape == (100, 100)

    # ========== Test dict ==========
    obj = {"obs": torch.rand(100, 100, dtype=torch.float32)}
    shm_obj = loader._create_shm_buffer(obj)
    assert isinstance(shm_obj.buf["obs"], ShmBuffer)

    payload = RecvPayload(proc_id=0, data=obj)
    loader._shm_callback(payload=payload, shm_obj=shm_obj)
    assert payload.data["obs"] is None

    loader._shm_putback(payload=payload, shm_obj=shm_obj)
    assert isinstance(payload.data["obs"], torch.Tensor)
    assert payload.data["obs"].shape == (100, 100)

    # ========== Test treetensor ==========
    obj = {"trajectories": [ttorch.as_tensor({"obs": torch.rand(10, 10, dtype=torch.float32)}) for _ in range(10)]}
    shm_obj = loader._create_shm_buffer(obj)

    payload = RecvPayload(proc_id=0, data=obj)
    loader._shm_callback(payload=payload, shm_obj=shm_obj)
    assert len(payload.data["trajectories"]) == 10
    for traj in payload.data["trajectories"]:
        assert traj["obs"] is None

    loader._shm_putback(payload=payload, shm_obj=shm_obj)
    for traj in payload.data["trajectories"]:
        assert isinstance(traj["obs"], torch.Tensor)
        assert traj["obs"].shape == (10, 10)


@pytest.mark.benchmark
def test_shared_object_benchmark():
    loader = FileStorageLoader(dirname="")
    # ========== Test treetensor ==========
    obj = {
        "env_step": 0,
        "trajectories": [
            ttorch.as_tensor(
                {
                    "done": False,
                    "reward": torch.tensor([1, 0], dtype=torch.int32),
                    "obs": torch.rand(4, 84, 84, dtype=torch.float32),
                    "next_obs": torch.rand(4, 84, 84, dtype=torch.float32),
                    "action": torch.tensor([1], dtype=torch.int32),
                    "collect_train_iter": torch.tensor([1], dtype=torch.int32),
                    "env_data_id": torch.tensor([1], dtype=torch.int32),
                }
            ) for _ in range(10)
        ]
    }
    buf = loader._create_shm_buffer(obj)
    payload = RecvPayload(proc_id=0, data=obj)
    loader._shm_callback(payload=payload, shm_obj=buf)

    def stmt():
        payload.extra = buf.id_.get()
        loader._shm_putback(payload=payload, shm_obj=buf)

    res = timeit.repeat(stmt, repeat=5, number=1000)
    print("Mean: {:.4f}s, STD: {:.4f}s, Mean each call: {:.4f}ms".format(np.mean(res), np.std(res), np.mean(res)))
    assert np.mean(res) < 1
