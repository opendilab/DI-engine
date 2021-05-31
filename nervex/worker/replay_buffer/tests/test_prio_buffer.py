import copy
from collections import defaultdict
import numpy as np
import pytest
from easydict import EasyDict
import os
import pickle

from nervex.worker.replay_buffer import PrioritizedReplayBuffer
from nervex.utils import deep_merge_dicts
from nervex.worker.replay_buffer.tests.conftest import generate_data, generate_data_list

demo_data_path = "test_demo_data.pkl"


@pytest.fixture(scope="function")
def setup_demo_buffer_factory():
    demo_data = {'data': [generate_data() for _ in range(10)]}
    with open(demo_data_path, "wb") as f:
        pickle.dump(demo_data, f)

    def generator():
        while True:
            cfg = copy.deepcopy(PrioritizedReplayBuffer.default_config())
            cfg.replay_buffer_size = 64
            cfg.max_use = 2
            cfg.max_staleness = 1000
            cfg.alpha = 0.6
            cfg.beta = 0.6
            cfg.enable_track_used_data = True
            demo_buffer = PrioritizedReplayBuffer(name="demo", cfg=cfg)
            yield demo_buffer

    return generator()


@pytest.mark.unittest
class TestBaseBuffer:

    def test_push(self):
        buffer_cfg = deep_merge_dicts(PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        start_pointer = prioritized_buffer._tail
        start_vaildlen = prioritized_buffer.validlen
        start_data_id = prioritized_buffer._next_unique_id
        valid_count = 0
        for _ in range(100):
            if prioritized_buffer._data[prioritized_buffer._tail] is None:
                valid_count += 1
            prioritized_buffer.push(generate_data(), 0)
        assert (prioritized_buffer.replay_buffer_size == 64)
        assert (prioritized_buffer.count() == 64)
        assert (prioritized_buffer.validlen == start_vaildlen + valid_count)
        assert (prioritized_buffer.push_count == start_vaildlen + 100)
        assert (prioritized_buffer._tail == (start_pointer + 100) % prioritized_buffer.replay_buffer_size)
        assert (prioritized_buffer._next_unique_id == start_data_id + 100)
        # invalid item append test
        prioritized_buffer.push([], 0)
        assert (prioritized_buffer.validlen == start_vaildlen + valid_count)
        assert (prioritized_buffer.push_count == start_vaildlen + 100)
        assert (prioritized_buffer._tail == (start_pointer + 100) % prioritized_buffer.replay_buffer_size)
        assert (prioritized_buffer._next_unique_id == start_data_id + 100)

        buffer_cfg = deep_merge_dicts(PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        start_pointer = prioritized_buffer._tail
        start_data_id = prioritized_buffer._next_unique_id
        replay_buffer_size = prioritized_buffer.replay_buffer_size
        extend_num = int(0.6 * replay_buffer_size)
        for i in range(1, 4):
            data = generate_data_list(extend_num)
            prioritized_buffer.push(data, 0)
            assert prioritized_buffer._tail == (start_pointer + extend_num * i) % replay_buffer_size
            assert prioritized_buffer._next_unique_id == start_data_id + extend_num * i
            assert prioritized_buffer._valid_count == min(start_data_id + extend_num * i, replay_buffer_size)

    def test_update(self):
        buffer_cfg = deep_merge_dicts(PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        for _ in range(64):
            prioritized_buffer.push(generate_data(), 0)
            assert prioritized_buffer.validlen == sum([d is not None for d in prioritized_buffer._data])
        selected_idx = [1, 4, 8, 30, 63]
        info = {'priority': [], 'replay_unique_id': [], 'replay_buffer_idx': []}
        for idx in selected_idx:
            info['priority'].append(np.random.uniform() + 64 - idx)
            info['replay_unique_id'].append(prioritized_buffer._data[idx]['replay_unique_id'])
            info['replay_buffer_idx'].append(prioritized_buffer._data[idx]['replay_buffer_idx'])

        for _ in range(8):
            prioritized_buffer.push(generate_data(), 0)
        origin_data = copy.deepcopy(prioritized_buffer._data)
        prioritized_buffer.update(info)
        assert (np.argmax(info['priority']) == 0)
        assert (prioritized_buffer._max_priority == max(info['priority'][2:]))
        assert (prioritized_buffer._max_priority != max(info['priority']))
        for i in range(2):
            assert (origin_data[selected_idx[i]]['priority'] == prioritized_buffer._data[selected_idx[i]]['priority'])
        eps = prioritized_buffer._eps
        for i in range(2, 5):
            assert (info['priority'][i] + eps == prioritized_buffer._data[selected_idx[i]]['priority'])

        # test beta
        prioritized_buffer.beta = 1.
        assert (prioritized_buffer.beta == 1.)

    def test_sample(self):
        buffer_cfg = deep_merge_dicts(
            PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=2))
        )
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        for _ in range(64):
            data = generate_data()
            data['priority'] = None
            prioritized_buffer.push(data, 0)
        use_dict = defaultdict(int)
        while True:
            can_sample = prioritized_buffer._sample_check(32, 0)
            if not can_sample:
                break
            batch = prioritized_buffer.sample(32, 0)
            assert (len(batch) == 32)
            assert (all([b['IS'] == 1.0 for b in batch])), [b['IS'] for b in batch]  # because priority is not updated
            idx = [b['replay_buffer_idx'] for b in batch]
            for i in idx:
                use_dict[i] += 1
        assert sum(
            map(lambda x: x[1] >= prioritized_buffer._max_use, use_dict.items())
        ) == prioritized_buffer.replay_buffer_size - prioritized_buffer.validlen
        for k, v in use_dict.items():
            if v > prioritized_buffer._max_use:
                assert prioritized_buffer._data[k] is None

    def test_head_tail(self):
        buffer_cfg = deep_merge_dicts(
            PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=4))
        )
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        for i in range(65):
            prioritized_buffer.push(generate_data(), 0)
        assert prioritized_buffer._head == prioritized_buffer._tail == 1
        info = {'replay_unique_id': [], 'replay_buffer_idx': [], 'priority': []}
        for data in prioritized_buffer._data:
            info['replay_unique_id'].append(data['replay_unique_id'])
            info['replay_buffer_idx'].append(data['replay_buffer_idx'])
            info['priority'].append(0.)
        info['priority'][1] = 1000.
        prioritized_buffer.update(info)
        while prioritized_buffer._data[1] is not None:
            data = prioritized_buffer.sample(1, 0)
            print(data)
        prioritized_buffer.push({'data_id': '1096'}, 0)
        assert prioritized_buffer._tail == 2
        assert prioritized_buffer._head == 2

    def test_weight(self):
        buffer_cfg = deep_merge_dicts(
            PrioritizedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=1))
        )
        prioritized_buffer = PrioritizedReplayBuffer('test', buffer_cfg)
        assert (prioritized_buffer.validlen == 0)  # assert empty buffer

        def get_weights(data_):
            weights_ = []
            for d in data_:
                if 'priority' not in d.keys() or d['priority'] is None:
                    weights_.append(prioritized_buffer.max_priority)
                else:
                    weights_.append(d['priority'])
            weights_ = np.array(weights_)
            weights_ = weights_ ** prioritized_buffer.alpha
            return weights_

        # first part(20 elements, smaller than buffer.replay_buffer_size)
        data = generate_data_list(20)
        prioritized_buffer.push(data, 0)

        assert (prioritized_buffer.replay_buffer_size == 64)
        assert (prioritized_buffer.beta == 0.4)
        assert (prioritized_buffer.alpha == 0.6)
        assert (hasattr(prioritized_buffer, '_sum_tree'))
        assert (hasattr(prioritized_buffer, '_min_tree'))
        assert (prioritized_buffer.validlen == 20)

        # tree test
        weights = get_weights(data)
        assert (np.fabs(weights.sum() - prioritized_buffer._sum_tree.reduce()) < 1e-6)

        # second part(80 elements, bigger than buffer.replay_buffer_size)
        data = generate_data_list(80)
        prioritized_buffer.push(data, 0)
        assert (prioritized_buffer.validlen == 64)
        assert (prioritized_buffer._next_unique_id == 20 + 80)
        assert (prioritized_buffer._tail == (20 + 80) % 64)
        weights = get_weights(data[-64:])
        assert (np.fabs(weights.sum() - prioritized_buffer._sum_tree.reduce()) < 1e-6)
        weights = get_weights(data[-36:])
        assert (np.fabs(weights.sum() - prioritized_buffer._sum_tree.reduce(start=0, end=36)) < 1e-6)


# @pytest.mark.unittest
class TestDemonstrationBuffer:

    def test_naive(self, setup_demo_buffer_factory):
        setup_demo_buffer = next(setup_demo_buffer_factory)
        naive_demo_buffer = next(setup_demo_buffer_factory)
        with open(demo_data_path, 'rb+') as f:
            data = pickle.load(f)
        setup_demo_buffer.load_state_dict(data)
        assert setup_demo_buffer.validlen == len(data['data'])  # assert buffer not empty
        samples = setup_demo_buffer.sample(3, 0)
        assert 'staleness' in samples[0]
        assert samples[1]['staleness'] == -1
        assert len(samples) == 3
        update_info = {'replay_unique_id': [0, 2], 'replay_buffer_idx': [0, 2], 'priority': [1.33, 1.44]}
        setup_demo_buffer.update(update_info)
        samples = setup_demo_buffer.sample(10, 0)
        for sample in samples:
            if sample['replay_unique_id'] == 0:
                assert abs(sample['priority'] - 1.33) <= 0.01 + 1e-5, sample
            if sample['replay_unique_id'] == 2:
                assert abs(sample['priority'] - 1.44) <= 0.02 + 1e-5, sample

        state_dict = setup_demo_buffer.state_dict()
        naive_demo_buffer.load_state_dict(state_dict)
        assert naive_demo_buffer._tail == setup_demo_buffer._tail
        assert naive_demo_buffer._max_priority == setup_demo_buffer._max_priority

        os.popen('rm -rf log')
        os.popen('rm -rf {}'.format(demo_data_path))
