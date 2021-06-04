import pytest
from easydict import EasyDict
import os
import time

from nervex.worker.replay_buffer import NaiveReplayBuffer
from nervex.utils import deep_merge_dicts
from nervex.worker.replay_buffer.tests.conftest import generate_data, generate_data_list


@pytest.mark.unittest
class TestNaiveBuffer:

    def test_push(self):
        buffer_cfg = deep_merge_dicts(NaiveReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        naive_buffer = NaiveReplayBuffer(buffer_cfg, name='test')
        start_pointer = naive_buffer._tail
        start_vaildlen = naive_buffer.count()
        valid_count = 0
        for _ in range(100):
            if naive_buffer._data[naive_buffer._tail] is None:
                valid_count += 1
            naive_buffer.push(generate_data(), 0)
        assert (naive_buffer.replay_buffer_size == 64)
        assert (naive_buffer.count() == 64 == start_vaildlen + valid_count)
        assert (naive_buffer.push_count == start_vaildlen + 100)
        assert (naive_buffer._tail == (start_pointer + 100) % naive_buffer.replay_buffer_size)
        naive_buffer.update({'no_info': True})

        buffer_cfg = deep_merge_dicts(NaiveReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        naive_buffer = NaiveReplayBuffer(buffer_cfg, name='test')
        start_pointer = naive_buffer._tail
        replay_buffer_size = naive_buffer.replay_buffer_size
        extend_num = int(0.6 * replay_buffer_size)
        for i in range(1, 4):
            data = generate_data_list(extend_num)
            naive_buffer.push(data, 0)
            assert naive_buffer._tail == (start_pointer + extend_num * i) % replay_buffer_size

    def test_sample(self):
        buffer_cfg = deep_merge_dicts(NaiveReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        naive_buffer = NaiveReplayBuffer(buffer_cfg, name='test')
        for _ in range(64):
            naive_buffer.push(generate_data(), 0)
        batch = naive_buffer.sample(32, 0)
        assert (len(batch) == 32)

        # test clear
        naive_buffer.clear()
        assert naive_buffer.count() == 0

    @pytest.mark.used
    def test_track_used_data(self):
        buffer_cfg = deep_merge_dicts(
            NaiveReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=10, enable_track_used_data=True))
        )
        naive_buffer = NaiveReplayBuffer(buffer_cfg, name='test')
        naive_buffer.start()

        old_data_list = generate_data_list(10, meta=True)
        naive_buffer.push(old_data_list, 0)
        for data in old_data_list:
            assert os.path.exists(data['data_id'])
        assert naive_buffer.count() == 10
        new_data_list = generate_data_list(8, meta=True)
        naive_buffer.push(new_data_list, 0)
        assert naive_buffer.count() == 10
        for data in new_data_list:
            assert os.path.exists(data['data_id'])
        time.sleep(1)
        for data in old_data_list[:8]:
            assert not os.path.exists(data['data_id'])
        naive_buffer.clear()
        time.sleep(1)
        for data in old_data_list[9:]:
            assert not os.path.exists(data['data_id'])
        for data in new_data_list:
            assert not os.path.exists(data['data_id'])

        naive_buffer.close()
