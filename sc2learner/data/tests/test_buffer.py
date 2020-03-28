import pytest
import copy
import numpy as np
from collections import defaultdict
from sc2learner.data import PrioritizedBuffer


@pytest.fixture(scope="function")
def setup_base_buffer():
    return PrioritizedBuffer(maxlen=64, max_reuse=2, min_sample_ratio=2., alpha=0., beta=0.)


@pytest.fixture(scope="function")
def setup_prioritized_buffer():
    return PrioritizedBuffer(maxlen=64, max_reuse=2, min_sample_ratio=2., alpha=0.6, beta=0.6)


def generate_data():
    ret = {
        'obs': np.random.randn(4),
    }
    p_weight = np.random.uniform()
    if p_weight < 1. / 3:
        pass  # no key 'priority'
    elif p_weight < 2. / 3:
        ret['priority'] = None
    else:
        ret['priority'] = np.random.uniform()

    return ret


class TestBaseBuffer:
    def test_append(self, setup_base_buffer):
        start_pointer = setup_base_buffer.pointer
        start_vaildlen = setup_base_buffer.validlen
        start_data_id = setup_base_buffer.data_id
        valid_count = 0
        for _ in range(100):
            if setup_base_buffer._data[setup_base_buffer.pointer] is None:
                valid_count += 1
            setup_base_buffer.append(generate_data())

        assert (setup_base_buffer.maxlen == 64)
        assert (setup_base_buffer.beta == 0.)
        assert (setup_base_buffer.alpha == 0.)
        assert (not setup_base_buffer.use_priority)
        assert (not hasattr(setup_base_buffer, 'min_tree'))
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.pointer == (start_pointer + 100) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.data_id == start_data_id + 100)

        # invalid item append test
        setup_base_buffer.append([])
        assert (setup_base_buffer.validlen == start_vaildlen + valid_count)
        assert (setup_base_buffer.pointer == (start_pointer + 100) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.data_id == start_data_id + 100)

        # tree test(alpha=0.)
        assert (np.fabs(setup_base_buffer.maxlen - setup_base_buffer.sum_tree.reduce()) < 1e-6)

    def test_extend(self, setup_base_buffer):
        start_pointer = setup_base_buffer.pointer
        start_vaildlen = setup_base_buffer.validlen
        start_data_id = setup_base_buffer.data_id

        data = []
        L = int(1.5 * setup_base_buffer.maxlen)
        for _ in range(L):
            data.append(generate_data())
        invalid_idx = np.random.choice([i for i in range(L)], int(0.1 * L), replace=False)
        for i in invalid_idx:
            data[i] = None

        setup_base_buffer.extend(data)
        valid_data_num = L - int(0.1 * L)
        assert (setup_base_buffer.pointer == (start_pointer + valid_data_num) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.data_id == start_data_id + valid_data_num)

        data = [None for _ in range(10)]
        setup_base_buffer.extend(data)
        assert (setup_base_buffer.pointer == (start_pointer + valid_data_num) % setup_base_buffer.maxlen)
        assert (setup_base_buffer.data_id == start_data_id + valid_data_num)
        assert (sum(setup_base_buffer._reuse_count) == 0)

    def test_beta(self, setup_base_buffer):
        assert (setup_base_buffer.beta == 0.)
        setup_base_buffer.beta = 1.
        assert (setup_base_buffer.beta == 1.)

    def test_update(self, setup_base_buffer):
        for _ in range(64):
            setup_base_buffer.append(generate_data())
        selected_idx = [1, 4, 8, 30, 63]
        info = {'priority': [], 'replay_buffer_id': [], 'replay_buffer_idx': []}
        for idx in selected_idx:
            info['priority'].append(np.random.uniform() + 64 - idx)
            info['replay_buffer_id'].append(setup_base_buffer._data[idx]['replay_buffer_id'])
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
        for i in range(2, 5):
            assert (info['priority'][i] == setup_base_buffer._data[selected_idx[i]]['priority'])

    def test_sample(self, setup_base_buffer):
        for _ in range(64):
            setup_base_buffer.append(generate_data())
        reuse_dict = defaultdict(int)
        while True:
            batch = setup_base_buffer.sample(32)
            if batch is None:
                break
            assert (len(batch) == 32)
            assert (all([b['IS'] == 1.0 for b in batch]))
            idx = [b['replay_buffer_idx'] for b in batch]
            for i in idx:
                reuse_dict[i] += 1
        assert sum(
            map(lambda x: x[1] > setup_base_buffer.max_reuse, reuse_dict.items())
        ) == setup_base_buffer.maxlen - setup_base_buffer.validlen
        for k, v in reuse_dict.items():
            if v > setup_base_buffer.max_reuse:
                assert setup_base_buffer._data[k] is None


class TestPrioritizedBuffer:
    def test_append(self, setup_prioritized_buffer):
        assert (setup_prioritized_buffer.validlen == 0)  # assert empty buffer

        def get_weights(data):
            weights = []
            for d in data:
                if 'priority' not in d.keys() or d['priority'] is None:
                    weights.append(setup_prioritized_buffer.max_priority)
                else:
                    weights.append(d['priority'])
            weights = np.array(weights)
            weights = weights**setup_prioritized_buffer.alpha
            return weights

        # first part(20 elements, which is smaller than buffer.maxlen)
        data = []
        for _ in range(20):
            tmp = generate_data()
            data.append(tmp)
            setup_prioritized_buffer.append(tmp)

        assert (setup_prioritized_buffer.maxlen == 64)
        assert (setup_prioritized_buffer.beta == 0.6)
        assert (setup_prioritized_buffer.alpha == 0.6)
        assert (setup_prioritized_buffer.use_priority)
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
        assert (setup_prioritized_buffer.data_id == 20 + 80)
        assert (setup_prioritized_buffer.pointer == (20 + 80) % 64)
        weights = get_weights(data[-64:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce()) < 1e-6)
        weights = get_weights(data[-36:])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce(start=0, end=36)) < 1e-6)
        weights = get_weights(data[36:64])
        assert (np.fabs(weights.sum() - setup_prioritized_buffer.sum_tree.reduce(start=36)) < 1e-6)
