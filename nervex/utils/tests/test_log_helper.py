import random
from collections import deque

import numpy as np
import pytest

from nervex.utils import AverageMeter


@pytest.mark.unittest
class TestAverageMeter:
    def test_naive(self):
        handle = AverageMeter(length=1)
        handle.reset()
        assert handle.val == 0.0
        assert handle.avg == 0.0
        for _ in range(10):
            t = random.uniform(0, 1)
            handle.update(t)
            assert handle.val == t
            assert handle.avg == pytest.approx(t, abs=1e-6)

        handle = AverageMeter(length=5)
        handle.reset()
        assert handle.val == 0.0
        assert handle.avg == 0.0
        queue = deque(maxlen=5)
        for _ in range(10):
            t = random.uniform(0, 1)
            handle.update(t)
            queue.append(t)
            assert handle.val == t
            assert handle.avg == pytest.approx(np.mean(queue, axis=0))
