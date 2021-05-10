import copy
from collections import defaultdict
import numpy as np
import pytest
from easydict import EasyDict
import os
import pickle

from nervex.data import NaiveReplayBuffer


@pytest.fixture(scope="function")
def setup_naive_buffer():
    return NaiveReplayBuffer(name="agent", cfg=EasyDict(dict(replay_buffer_size=64)))


def generate_data():
    return {'obs': np.random.randn(4), 'data_id': 0}


@pytest.mark.unittest
class TestBaseBuffer:

    def test_append(self, setup_naive_buffer):
        start_pointer = setup_naive_buffer._tail
        start_vaildlen = setup_naive_buffer.validlen
        start_data_id = setup_naive_buffer._next_unique_id
        valid_count = 0
        for _ in range(100):
            if setup_naive_buffer._data[setup_naive_buffer._tail] is None:
                valid_count += 1
            setup_naive_buffer.append(generate_data())

        assert (setup_naive_buffer.replay_buffer_size == 64)
        assert (setup_naive_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_naive_buffer.push_count == start_vaildlen + 100)
        assert (setup_naive_buffer._tail == (start_pointer + 100) % setup_naive_buffer.replay_buffer_size)
        assert (setup_naive_buffer._next_unique_id == start_data_id + 100)

        del setup_naive_buffer

    def test_extend(self, setup_naive_buffer):
        start_pointer = setup_naive_buffer._tail
        start_data_id = setup_naive_buffer._next_unique_id
        replay_buffer_size = setup_naive_buffer.replay_buffer_size

        extend_num = int(0.6 * replay_buffer_size)
        for i in range(1, 4):
            data = [generate_data() for _ in range(extend_num)]
            setup_naive_buffer.extend(data)
            assert setup_naive_buffer._tail == (start_pointer + extend_num * i) % replay_buffer_size
            assert setup_naive_buffer._next_unique_id == start_data_id + extend_num * i
            assert setup_naive_buffer._valid_count == min(start_data_id + extend_num * i, replay_buffer_size)

    def test_sample(self, setup_naive_buffer):
        for _ in range(64):
            setup_naive_buffer.append(generate_data())
        can_sample = setup_naive_buffer.sample_check(32, 0)
        assert can_sample
        batch = setup_naive_buffer.sample(32, 0)
        assert (len(batch) == 32)
        idx_list = [b['replay_buffer_idx'] for b in batch]
        idx_set = set(idx_list)
        assert len(idx_list) == len(idx_set)
