import pytest
import time
import numpy as np
import torch
from multiprocessing import Process

from ding.envs.env_manager.subprocess_env_manager import ShmBuffer


def writer(shm):
    while True:
        shm.fill(np.random.random(size=(4, 84, 84)).astype(np.float32))
        time.sleep(1)


@pytest.mark.unittest
def test_shm():

    shm = ShmBuffer(dtype=np.float32, shape=(4, 84, 84), copy_on_get=False)
    writer_process = Process(target=writer, args=(shm, ))
    writer_process.start()

    time.sleep(0.1)

    data1 = shm.get()
    time.sleep(1)
    data2 = shm.get()
    # same memory
    assert (data1 == data2).all()

    time.sleep(1)
    data3 = shm.get().copy()
    time.sleep(1)
    data4 = shm.get()
    assert (data3 != data4).all()

    writer_process.terminate()
