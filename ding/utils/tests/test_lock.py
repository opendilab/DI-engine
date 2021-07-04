import pytest
import numpy as np
from collections import deque

from ding.utils import LockContext, LockContextType, get_rw_file_lock


@pytest.mark.unittest
def test_usage():
    lock = LockContext(LockContextType.PROCESS_LOCK)
    queue = deque(maxlen=10)
    data = np.random.randn(4)
    with lock:
        queue.append(np.copy(data))
    with lock:
        output = queue.popleft()
    assert (output == data).all()
    lock.acquire()
    queue.append(np.copy(data))
    lock.release()
    lock.acquire()
    output = queue.popleft()
    lock.release()
    assert (output == data).all()


@pytest.mark.unittest
def test_get_rw_file_lock():
    path = 'tmp.npy'
    # TODO real read-write case
    read_lock = get_rw_file_lock(path, 'read')
    write_lock = get_rw_file_lock(path, 'write')
    with write_lock:
        np.save(path, np.random.randint(0, 1, size=(3, 4)))
    with read_lock:
        data = np.load(path)
    assert data.shape == (3, 4)
