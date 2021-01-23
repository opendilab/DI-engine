import copy
from collections import defaultdict
import numpy as np
import pytest
from easydict import EasyDict
import os
import pickle

from nervex.data import ReplayBuffer

monitor_cfg = EasyDict(
    {
        'log_freq': 2000,
        'log_path': './log/buffer/a_buffer/',
        'natural_expire': 100,
        'tick_expire': 100,
    }
)


@pytest.fixture(scope="function")
def setup_base_buffer():
    return ReplayBuffer(
        name="agent", maxlen=64, max_reuse=2, min_sample_ratio=2., alpha=0., beta=0., monitor_cfg=monitor_cfg
    )


@pytest.fixture(scope="function")
def setup_prioritized_buffer():
    return ReplayBuffer(
        name="agent",
        maxlen=64,
        max_reuse=2,
        min_sample_ratio=2.,
        alpha=0.6,
        beta=0.6,
        enable_track_used_data=True,
        monitor_cfg=monitor_cfg
    )


@pytest.fixture(scope="function")
def setup_demo_buffer():
    demo_data_list = [generate_data() for _ in range(10)]
    with open("test_demo_data.pkl", "wb") as f:
        pickle.dump(demo_data_list, f)
    demo_buffer = ReplayBuffer(
        name="demo",
        load_path="test_demo_data.pkl",
        maxlen=64,
        max_reuse=2,
        min_sample_ratio=1.,
        alpha=0.6,
        beta=0.6,
        enable_track_used_data=True,
        monitor_cfg=monitor_cfg
    )
    os.popen("rm -rf test_demo_data.pkl")
    return demo_buffer


def generate_data():
    ret = {'obs': np.random.randn(4)}
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
        start_pointer = setup_base_buffer.pointer
        start_vaildlen = setup_base_buffer.validlen
        start_data_id = setup_base_buffer.next_unique_id
        valid_count = 0
        for _ in range(100):
            if setup_base_buffer._data[setup_base_buffer.pointer] is None:
                valid_count += 1
            setup_base_buffer.append(generate_data())

        assert (setup_base_buffer.maxlen == 64)
        assert (setup_base_buffer.beta == 0.)
        assert (setup_base_buffer.alpha == 0.)
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.push_count == start_vaildlen + 100)
        assert (setup_base_buffer.pointer == (start_pointer + 100) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.next_unique_id == start_data_id + 100)

        # invalid item append test
        setup_base_buffer.append([])
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.push_count == start_vaildlen + 100)
        assert (setup_base_buffer.pointer == (start_pointer + 100) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.next_unique_id == start_data_id + 100)

    def test_extend(self, setup_base_buffer):
        start_pointer = setup_base_buffer.pointer
        start_data_id = setup_base_buffer.next_unique_id

        init_num = int(0.2 * setup_base_buffer.maxlen)
        data = [generate_data() for _ in range(init_num)]
        setup_base_buffer.extend(data)
        assert setup_base_buffer.pointer == start_pointer + init_num
        assert setup_base_buffer.next_unique_id == start_data_id + init_num
        start_pointer += init_num
        start_data_id += init_num

        data = []
        enlarged_length = int(1.5 * setup_base_buffer.maxlen)
        for _ in range(enlarged_length):
            data.append(generate_data())
        invalid_idx = np.random.choice([i for i in range(enlarged_length)], int(0.1 * enlarged_length), replace=False)
        for i in invalid_idx:
            data[i] = None

        setup_base_buffer.extend(data)
        valid_data_num = enlarged_length - int(0.1 * enlarged_length)
        assert setup_base_buffer.pointer == (start_pointer + valid_data_num) % setup_base_buffer.maxlen
        assert setup_base_buffer.next_unique_id == start_data_id + valid_data_num

        data = [None for _ in range(10)]
        setup_base_buffer.extend(data)
        assert setup_base_buffer.pointer == (start_pointer + valid_data_num) % setup_base_buffer.maxlen
        assert setup_base_buffer.next_unique_id == start_data_id + valid_data_num
        assert sum(setup_base_buffer._reuse_count.values()) == 0, sum(setup_base_buffer._reuse_count)

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
        assert (setup_base_buffer.max_priority == max(info['priority'][2:]))
        assert (setup_base_buffer.max_priority != max(info['priority']))
        for i in range(2):
            assert (origin_data[selected_idx[i]]['priority'] == setup_base_buffer._data[selected_idx[i]]['priority'])
        eps = setup_base_buffer._eps
        for i in range(2, 5):
            assert (info['priority'][i] + eps == setup_base_buffer._data[selected_idx[i]]['priority'])

    def test_sample(self, setup_base_buffer):
        for _ in range(64):
            setup_base_buffer.append(generate_data())
        reuse_dict = defaultdict(int)
        while True:
            can_sample = setup_base_buffer.sample_check(32, 0)
            if not can_sample:
                break
            batch = setup_base_buffer.sample(32, 0)
            assert (len(batch) == 32)
            assert (all([b['IS'] == 1.0 for b in batch]))  # because priority is not updated
            idx = [b['replay_buffer_idx'] for b in batch]
            for i in idx:
                reuse_dict[i] += 1
        assert sum(
            map(lambda x: x[1] > setup_base_buffer.max_reuse, reuse_dict.items())
        ) == setup_base_buffer.maxlen - setup_base_buffer.validlen
        for k, v in reuse_dict.items():
            if v > setup_base_buffer.max_reuse:
                assert setup_base_buffer._data[k] is None
        assert setup_base_buffer.used_data is None

        os.popen('rm -rf log')


@pytest.mark.unittest
class TestReplayBuffer:

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

        # first part(20 elements, which is smaller than buffer.maxlen)
        data = []
        for _ in range(20):
            tmp = generate_data()
            data.append(tmp)
            setup_prioritized_buffer.append(tmp)

        assert (setup_prioritized_buffer.maxlen == 64)
        assert (setup_prioritized_buffer.beta == 0.6)
        assert (setup_prioritized_buffer.alpha == 0.6)
        assert (hasattr(setup_prioritized_buffer, 'sum_tree'))
        assert (hasattr(setup_prioritized_buffer, 'min_tree'))
        assert (setup_prioritized_buffer.validlen == 20)

        # tree test
        weights = get_weights(data)
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce()) < 1e-6)

        # second part(80 elements, which is bigger than buffer.maxlen)
        for _ in range(80):
            tmp = generate_data()
            data.append(tmp)
            setup_prioritized_buffer.append(tmp)
        assert (setup_prioritized_buffer.validlen == 64)
        assert (setup_prioritized_buffer.next_unique_id == 20 + 80)
        assert (setup_prioritized_buffer.pointer == (20 + 80) % 64)
        weights = get_weights(data[-64:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce()) < 1e-6)
        weights = get_weights(data[-36:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce(start=0, end=36)) < 1e-6)
        weights = get_weights(data[36:64])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce(start=36)) < 1e-6)

    def test_used_data(self, setup_prioritized_buffer):
        for _ in range(setup_prioritized_buffer._maxlen + 2):
            setup_prioritized_buffer.append({})
        setup_prioritized_buffer.extend([{}])
        for _ in range(2 + 1):
            assert setup_prioritized_buffer.used_data is not None
        assert setup_prioritized_buffer.used_data is None
        os.popen('rm -rf log')


@pytest.mark.unittest
class TestDemonstrationBuffer:

    def test_naive(self, setup_demo_buffer):
        assert setup_demo_buffer.validlen != 0  # assert buffer not empty
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
        os.popen('rm -rf log')
