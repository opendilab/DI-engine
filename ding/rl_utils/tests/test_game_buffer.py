import pytest
import time
from typing import Callable
from ding.rl_utils.mcts.game_buffer import GameBuffer
from ding.data.buffer.buffer import BufferedData
import numpy as np
from easydict import EasyDict
from ding.torch_utils import to_list


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


config = EasyDict(dict(
    batch_size=10,
    transition_num=20,
    priority_prob_alpha=0.5,
    total_transitions=10000,
))


@pytest.mark.unittest
def test_naive_push_sample():
    buffer = GameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    for i in range(20):
        buffer.push(to_list(np.multiply(i, data)), meta)
    assert buffer.count() == 20

    # push games
    buffer.push_games([data, data], [meta, meta])
    assert buffer.count() == 22

    # Clear
    buffer.clear()
    assert buffer.count() == 0

    # sample
    for i in range(5):
        buffer.push(to_list(np.multiply(i, data)), meta)

    # assert len(buffer.sample(indices=['0', '1', '2', '3', '4'])) == 5
    assert len(buffer.sample(indices=[0, 1, 2, 3, 4])) == 5

    assert len(buffer.sample(size=5)) == 5


@pytest.mark.unittest
def test_update():
    buffer = GameBuffer(config)
    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    for i in range(20):
        buffer.push(to_list(np.multiply(i, data)), meta)
    assert buffer.count() == 20

    # update
    meta_new = {'priorities': 0.999}
    buffer.update(0, data, meta_new)
    print(buffer.sample(indices=[0]))
    assert buffer.priorities[0] == 0.999

    assert buffer.update(200, data, meta_new) is False


@pytest.mark.unittest
def test_rate_limit_push_sample():
    buffer = GameBuffer(config).use(RateLimit(max_rate=5))
    # fake data
    data = [[1, 1, 1] for i in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    for i in range(20):
        buffer.push(to_list(np.multiply(i, data)), meta)
    # print(buffer.sample(indices=[0,1,2,3,4]))
    # print(buffer.sample(5))

    assert buffer.count() == 20


@pytest.mark.unittest
def test_prepare_batch_context():
    buffer = GameBuffer(config)

    # fake data
    data_1 = [[1, 1, 1] for i in range(10)]  # (s,a,r)
    meta_1 = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    data_2 = [[1, 1, 1] for i in range(10, 20)]  # (s,a,r)
    meta_2 = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    buffer.push(data_1, meta_1)
    buffer.push(data_2, meta_2)

    context = buffer.prepare_batch_context(batch_size=2, beta=0.2)
    # context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
    # print(context)


@pytest.mark.unittest
def test_buffer_view():
    buf1 = GameBuffer(config)

    # fake data
    data = [[1, 1, 1] for _ in range(10)]  # (s,a,r)
    meta = {'end_tag': True, 'gap_steps': 5, 'priorities': np.array([0.9 for i in range(10)])}

    # push
    buf1.push(data, meta)
    assert buf1.count() == 1

    buf2 = buf1.view()

    for i in range(10):
        buf2.push(to_list(np.multiply(i, data)), meta)

    assert buf1.count() == 1
    assert buf2.count() == 10
