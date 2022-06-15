from time import sleep, time
import pytest
import numpy as np
import timeit
import pickle
import multiprocessing as mp
from ding.utils import Profiler

from ding.envs.env_manager.subprocess_env_manager import ShmBufferContainer

data = np.random.random((3600, 300)).astype(np.float32)
data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def only_dump():
    pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def only_load():
    pickle.loads(data_bytes)


def dump_load():
    s = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.loads(s)


def loader_process(shm: ShmBufferContainer, send_q, recv_q):
    buffer = []
    start = time()
    for _ in range(600):
        buffer.append(pickle.loads(data_bytes))
    print("Loaded: {:.2f}".format(time() - start))
    while True:
        send_q.get()
        if len(buffer) == 0:
            recv_q.put(1)
            break
        shm.fill(buffer.pop(0))
        recv_q.put(0)


def only_load_shared_memory():
    ctx = mp.get_context()
    shm = ShmBufferContainer(data.dtype, data.shape, copy_on_get=False)

    send_q = ctx.Queue()
    recv_q = ctx.Queue()

    loader = ctx.Process(target=loader_process, args=(shm, send_q, recv_q))
    loader.start()

    send_q.put(0)
    start = None
    i = 0
    while True:
        i += 1
        flag = recv_q.get()
        if start is None:
            start = time()
        if flag == 1:
            break
        d = shm.get()
        send_q.put(0)
    print("Time cost: {:.3f}, times: {}".format(time() - start, i))

    loader.join()


def summary(name, res):
    template = "Task Name: {} >>> Mean: {:.3f}s, STD: {:.3f}s"
    print(template.format(name, np.mean(res), np.std(res)))


@pytest.mark.benchmark
def test_data_serializer_benchmark():
    print("=========== test_data_serializer_benchmark ===========")
    # Profiler().profile("./tmp/shared_memory")
    start = time()
    for _ in range(600):
        only_load()
    print("Only load: {:.3f}".format(time() - start))

    # # summary("Only Dump", timeit.repeat(only_dump, repeat=3, number=50))
    # # summary("Only Load", timeit.repeat(only_load, repeat=3, number=50))
    # # summary("Dump Load", timeit.repeat(dump_load, repeat=3, number=50))
    only_load_shared_memory()
