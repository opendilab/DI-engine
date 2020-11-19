import pytest
import numpy as np
from collections import deque

from nervex.utils import LockContext, LockContextType


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
