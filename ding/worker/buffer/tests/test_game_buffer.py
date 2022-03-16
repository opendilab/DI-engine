import pytest
import time
import random
from typing import Callable
from ding.worker.buffer.game_buffer import GameBuffer
from ding.worker.buffer.buffer import BufferedData
from torch.utils.data import DataLoader
import numpy as np

class RateLimit:
    r"""
    Add rate limit threshold to push function
    """

    def __init__(self, max_rate: int = float("inf"), window_seconds: int = 30) -> None:
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.buffered = []

    def __call__(self, action: str, chain: Callable, *args, **kwargs):
        if action == "push":
            return self.push(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    def push(self, chain, data, *args, **kwargs) -> None:
        current = time.time()
        # Cut off stale records
        self.buffered = [t for t in self.buffered if t > current - self.window_seconds]
        if len(self.buffered) < self.max_rate:
            self.buffered.append(current)
            return chain(data, *args, **kwargs)
        else:
            return None


def add_10() -> Callable:
    """
    Transform data on sampling
    """

    def sample(chain: Callable, size: int, replace: bool = False, *args, **kwargs):
        sampled_data = chain(size, replace, *args, **kwargs)
        return [BufferedData(data=item.data + 10, index=item.index, meta=item.meta) for item in sampled_data]

    def _subview(action: str, chain: Callable, *args, **kwargs):
        if action == "sample":
            return sample(chain, *args, **kwargs)
        return chain(*args, **kwargs)

    return _subview


@pytest.mark.unittest
def test_naive_push_sample():
    from easydict import EasyDict

    config = EasyDict(dict(
        batch_size=10,
        transition_num=20,
        priority_prob_alpha=0.5,
        total_transitions=10000,
    ))
    buffer = GameBuffer(config)

    # fake data
    data = [[2, 1, 0.5] for i in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    for i in range(20):
        buffer.push(data, meta)
    assert buffer.count() == 20
    print(buffer.sample_game(10))

    # push
    for i in range(5):
        buffer.push(data, meta)
    assert buffer.count() == 25

    # Clear
    buffer.clear()
    assert buffer.count() == 0


@pytest.mark.unittest
def test_prepare_batch_context():
    from easydict import EasyDict

    config = EasyDict(dict(
        batch_size=10,
        transition_num=20,
        priority_prob_alpha=0.5,
        total_transitions=10000,
    ))
    buffer = GameBuffer(config)

    # fake data
    data = [[2, 1, 0.5] for i in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    for i in range(20):
        buffer.push(data, meta)

    context = buffer.prepare_batch_context(batch_size=2, beta=0.2)
    # context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
    print(context)


