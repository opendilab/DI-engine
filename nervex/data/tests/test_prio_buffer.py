import copy
from collections import defaultdict
import numpy as np
import pytest
from easydict import EasyDict
import os
import pickle

from nervex.data import PrioritizedReplayBuffer

monitor_cfg = EasyDict(
    {
        'log_freq': 2000,
        'log_path': './log/buffer/a_buffer/',
        'natural_expire': 100,
        'tick_expire': 100,
    }
)
demo_data_path = "test_demo_data.pkl"


@pytest.fixture(scope="function")
def setup_base_buffer():
    cfg = copy.deepcopy(PrioritizedReplayBuffer.default_config())
    cfg.replay_buffer_size = 64
    cfg.max_use = 2
    cfg.min_sample_ratio = 2.
    cfg.alpha = 0.
    cfg.beta = 0.
    cfg.monitor = monitor_cfg

    return PrioritizedReplayBuffer(name="agent", cfg=cfg)


@pytest.fixture(scope="function")
def setup_prioritized_buffer():
    cfg = copy.deepcopy(PrioritizedReplayBuffer.default_config())
    cfg.replay_buffer_size = 64
    cfg.max_use = 1
    cfg.min_sample_ratio = 2.
    cfg.max_staleness = 1000
    cfg.enable_track_used_data = True
    cfg.monitor = monitor_cfg

    return PrioritizedReplayBuffer(name="agent", cfg=cfg)


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
            cfg.min_sample_ratio = 2.
            cfg.max_staleness = 1000
            cfg.alpha = 0.6
            cfg.beta = 0.6
            cfg.enable_track_used_data = True
            cfg.monitor_cfg = monitor_cfg
            demo_buffer = PrioritizedReplayBuffer(name="demo", cfg=cfg)
            yield demo_buffer

    return generator()


def generate_data():
    ret = {'obs': np.random.randn(4), 'data_id': 0}
    p_weight = np.random.uniform()
    if p_weight < 1. / 3:
        pass  # no key 'priority'
    elif p_weight < 2. / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform()
    return ret


@pytest.mark.unittest
class TestBaseBuffer:

    def test_append(self, setup_base_buffer):
        start_pointer = setup_base_buffer._tail
        start_vaildlen = setup_base_buffer.validlen
        start_data_id = setup_base_buffer._next_unique_id
        valid_count = 0
        for _ in range(100):
            if setup_base_buffer._data[setup_base_buffer._tail] is None:
                valid_count += 1
            setup_base_buffer.append(generate_data())

        assert (setup_base_buffer.replay_buffer_size == 64)
        assert (setup_base_buffer.beta == 0.)
        assert (setup_base_buffer.alpha == 0.)
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.push_count == start_vaildlen + 100)
        assert (setup_base_buffer._tail == (start_pointer + 100) % setup_base_buffer.replay_buffer_size)
        assert (setup_base_buffer._next_unique_id == start_data_id + 100)

        # invalid item append test
        setup_base_buffer.append([])
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.push_count == start_vaildlen + 100)
        assert (setup_base_buffer._tail == (start_pointer + 100) % setup_base_buffer.replay_buffer_size)
        assert (setup_base_buffer._next_unique_id == start_data_id + 100)

    def test_extend(self, setup_base_buffer):
        start_pointer = setup_base_buffer._tail
        start_data_id = setup_base_buffer._next_unique_id

        init_num = int(0.2 * setup_base_buffer.replay_buffer_size)
        data = [generate_data() for _ in range(init_num)]
        setup_base_buffer.extend(data)
        assert setup_base_buffer._tail == start_pointer + init_num
        assert setup_base_buffer._next_unique_id == start_data_id + init_num
        start_pointer += init_num
        start_data_id += init_num

        data = []
        enlarged_length = int(1.5 * setup_base_buffer.replay_buffer_size)
        for _ in range(enlarged_length):
            data.append(generate_data())
        invalid_idx = np.random.choice([i for i in range(enlarged_length)], int(0.1 * enlarged_length), replace=False)
        for i in invalid_idx:
            data[i] = None

        setup_base_buffer.extend(data)
        valid_data_num = enlarged_length - int(0.1 * enlarged_length)
        assert setup_base_buffer._tail == (start_pointer + valid_data_num) % setup_base_buffer.replay_buffer_size
        assert setup_base_buffer._next_unique_id == start_data_id + valid_data_num

        data = [None for _ in range(10)]
        setup_base_buffer.extend(data)
        assert setup_base_buffer._tail == (start_pointer + valid_data_num) % setup_base_buffer.replay_buffer_size
        assert setup_base_buffer._next_unique_id == start_data_id + valid_data_num
        assert sum(setup_base_buffer._use_count.values()) == 0, sum(setup_base_buffer._use_count)

    def test_beta(self, setup_base_buffer):
        assert (setup_base_buffer.beta == 0.)
        setup_base_buffer.beta = 1.
        assert (setup_base_buffer.beta == 1.)

    def test_update(self, setup_base_buffer):
        for _ in range(64):
            setup_base_buffer.append(generate_data())
            assert setup_base_buffer.validlen == sum([d is not None for d in setup_base_buffer._data])
        selected_idx = [1, 4, 8, 30, 63]
        info = {'priority': [], 'replay_unique_id': [], 'replay_buffer_idx': []}
        for idx in selected_idx:
            info['priority'].append(np.random.uniform() + 64 - idx)
            info['replay_unique_id'].append(setup_base_buffer._data[idx]['replay_unique_id'])
            info['replay_buffer_idx'].append(setup_base_buffer._data[idx]['replay_buffer_idx'])

        for _ in range(8):
            setup_base_buffer.append(generate_data())
        origin_data = copy.deepcopy(setup_base_buffer._data)
        setup_base_buffer.update(info)
        assert (np.argmax(info['priority']) == 0)
        assert (setup_base_buffer._max_priority == max(info['priority'][2:]))
        assert (setup_base_buffer._max_priority != max(info['priority']))
        for i in range(2):
            assert (origin_data[selected_idx[i]]['priority'] == setup_base_buffer._data[selected_idx[i]]['priority'])
        eps = setup_base_buffer._eps
        for i in range(2, 5):
            assert (info['priority'][i] + eps == setup_base_buffer._data[selected_idx[i]]['priority'])

    def test_sample(self, setup_base_buffer):
        for _ in range(64):
            setup_base_buffer.append(generate_data())
        use_dict = defaultdict(int)
        while True:
            can_sample = setup_base_buffer.sample_check(32, 0)
            if not can_sample:
                break
            batch = setup_base_buffer.sample(32, 0)
            assert (len(batch) == 32)
            assert (all([b['IS'] == 1.0 for b in batch]))  # because priority is not updated
            idx = [b['replay_buffer_idx'] for b in batch]
            for i in idx:
                use_dict[i] += 1
        assert sum(
            map(lambda x: x[1] >= setup_base_buffer._max_use, use_dict.items())
        ) == setup_base_buffer.replay_buffer_size - setup_base_buffer.validlen
        for k, v in use_dict.items():
            if v > setup_base_buffer._max_use:
                assert setup_base_buffer._data[k] is None
        assert setup_base_buffer.used_data is None


@pytest.mark.unittest
class TestPrioritizedReplayBuffer:

    def test_head_tail(self, setup_prioritized_buffer):
        for i in range(65):
            setup_prioritized_buffer.append(generate_data())
        assert setup_prioritized_buffer._head == setup_prioritized_buffer._tail == 1
        info = {'replay_unique_id': [], 'replay_buffer_idx': [], 'priority': []}
        for data in setup_prioritized_buffer._data:
            info['replay_unique_id'].append(data['replay_unique_id'])
            info['replay_buffer_idx'].append(data['replay_buffer_idx'])
            info['priority'].append(0.)
        info['priority'][1] = 1000.
        setup_prioritized_buffer.update(info)
        while setup_prioritized_buffer._data[1] is not None:
            data = setup_prioritized_buffer.sample(1, 0)
            print(data)
        setup_prioritized_buffer.append({'index': 1000})
        assert setup_prioritized_buffer._tail == 2
        assert setup_prioritized_buffer._head == 2

    def test_append(self, setup_prioritized_buffer):
        assert (setup_prioritized_buffer.validlen == 0)  # assert empty buffer

        def get_weights(data_):
            weights_ = []
            for d in data_:
                if 'priority' not in d.keys() or d['priority'] is None:
                    weights_.append(setup_prioritized_buffer.max_priority)
                else:
                    weights_.append(d['priority'])
            weights_ = np.array(weights_)
            weights_ = weights_ ** setup_prioritized_buffer.alpha
            return weights_

        # first part(20 elements, which is smaller than buffer.replay_buffer_size)
        data = []
        for _ in range(20):
            tmp = generate_data()
            data.append(tmp)
            setup_prioritized_buffer.append(tmp)

        assert (setup_prioritized_buffer.replay_buffer_size == 64)
        assert (setup_prioritized_buffer.beta == 0.4)
        assert (setup_prioritized_buffer.alpha == 0.6)
        assert (hasattr(setup_prioritized_buffer, '_sum_tree'))
        assert (hasattr(setup_prioritized_buffer, '_min_tree'))
        assert (setup_prioritized_buffer.validlen == 20)

        # tree test
        weights = get_weights(data)
        assert (np.fabs(weights.sum() - setup_prioritized_buffer._sum_tree.reduce()) < 1e-6)

        # second part(80 elements, which is bigger than buffer.replay_buffer_size)
        for _ in range(80):
            tmp = generate_data()
            data.append(tmp)
            setup_prioritized_buffer.append(tmp)
        assert (setup_prioritized_buffer.validlen == 64)
        assert (setup_prioritized_buffer._next_unique_id == 20 + 80)
        assert (setup_prioritized_buffer._tail == (20 + 80) % 64)
        weights = get_weights(data[-64:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer._sum_tree.reduce()) < 1e-6)
        weights = get_weights(data[-36:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer._sum_tree.reduce(start=0, end=36)) < 1e-6)
        weights = get_weights(data[36:64])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer._sum_tree.reduce(start=36)) < 1e-6)

    def test_used_data(self, setup_prioritized_buffer):
        for _ in range(setup_prioritized_buffer._replay_buffer_size + 2):
            setup_prioritized_buffer.append({'data_id': 0, 'collect_iter': 0})
        assert setup_prioritized_buffer._used_data.qsize() == 2
        setup_prioritized_buffer.extend([{'data_id': 0, 'collect_iter': 1}])
        assert setup_prioritized_buffer._used_data.qsize() == 3
        assert not setup_prioritized_buffer.sample_check(2, 1000)
        assert setup_prioritized_buffer._used_data.qsize() == setup_prioritized_buffer._replay_buffer_size + 2
        for _ in range(setup_prioritized_buffer._replay_buffer_size + 2):
            assert setup_prioritized_buffer.used_data is not None
        assert setup_prioritized_buffer.used_data is None
        for i in range(setup_prioritized_buffer._replay_buffer_size):
            setup_prioritized_buffer.append({'data_id': 'new_{}'.format(i), 'collect_iter': 2})
        assert setup_prioritized_buffer._used_data.qsize() == 1
        sampled_data = setup_prioritized_buffer.sample(16, 1001)
        assert len(sampled_data) == 16
        assert setup_prioritized_buffer._used_data.qsize() == 1
        assert len(setup_prioritized_buffer._using_data) == 16
        assert len(setup_prioritized_buffer._using_used_data) == 16
        sample_id_part = [d['data_id'] for d in sampled_data[:4]]
        setup_prioritized_buffer.update({'used_id': [i for i in sample_id_part]})
        assert setup_prioritized_buffer._used_data.qsize() == 5
        assert len(setup_prioritized_buffer._using_data) == 12
        assert len(setup_prioritized_buffer._using_used_data) == 12


@pytest.mark.unittest
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
