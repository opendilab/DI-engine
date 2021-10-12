import copy
from collections import defaultdict
import numpy as np
import pytest
from easydict import EasyDict
import os
import pickle
import time

from ding.worker.replay_buffer import AdvancedReplayBuffer
from ding.utils import deep_merge_dicts
from ding.worker.replay_buffer.tests.conftest import generate_data, generate_data_list

demo_data_path = "test_demo_data"


@pytest.fixture(scope="function")
def setup_demo_buffer_factory():
    demo_data = {'data': generate_data_list(10)}
    with open(demo_data_path, "wb") as f:
        pickle.dump(demo_data, f)

    def generator():
        while True:
            cfg = copy.deepcopy(AdvancedReplayBuffer.default_config())
            cfg.replay_buffer_size = 64
            cfg.max_use = 2
            cfg.max_staleness = 1000
            cfg.alpha = 0.6
            cfg.beta = 0.6
            cfg.enable_track_used_data = False
            yield AdvancedReplayBuffer(instance_name="demo", cfg=cfg)

    return generator()


@pytest.mark.unittest
class TestAdvancedBuffer:

    def test_push(self):
        buffer_cfg = deep_merge_dicts(AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        start_pointer = advanced_buffer._tail
        start_vaildlen = advanced_buffer.count()
        start_data_id = advanced_buffer._next_unique_id
        valid_count = 0
        for _ in range(100):
            if advanced_buffer._data[advanced_buffer._tail] is None:
                valid_count += 1
            advanced_buffer.push(generate_data(), 0)
        assert (advanced_buffer.replay_buffer_size == 64)
        assert (advanced_buffer.count() == 64 == start_vaildlen + valid_count)
        assert (advanced_buffer.push_count == start_vaildlen + 100)
        assert (advanced_buffer._tail == (start_pointer + 100) % advanced_buffer.replay_buffer_size)
        assert (advanced_buffer._next_unique_id == start_data_id + 100)
        # invalid item append test
        advanced_buffer.push([], 0)
        assert (advanced_buffer.count() == 64 == start_vaildlen + valid_count)
        assert (advanced_buffer.push_count == start_vaildlen + 100)
        assert (advanced_buffer._tail == (start_pointer + 100) % advanced_buffer.replay_buffer_size)
        assert (advanced_buffer._next_unique_id == start_data_id + 100)

        buffer_cfg = deep_merge_dicts(AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        start_pointer = advanced_buffer._tail
        start_data_id = advanced_buffer._next_unique_id
        replay_buffer_size = advanced_buffer.replay_buffer_size
        extend_num = int(0.6 * replay_buffer_size)
        for i in range(1, 4):
            data = generate_data_list(extend_num)
            advanced_buffer.push(data, 0)
            assert advanced_buffer._tail == (start_pointer + extend_num * i) % replay_buffer_size
            assert advanced_buffer._next_unique_id == start_data_id + extend_num * i
            assert advanced_buffer._valid_count == min(start_data_id + extend_num * i, replay_buffer_size)

    def test_update(self):
        buffer_cfg = deep_merge_dicts(AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64)))
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        for _ in range(64):
            advanced_buffer.push(generate_data(), 0)
            assert advanced_buffer.count() == sum([d is not None for d in advanced_buffer._data])
        selected_idx = [1, 4, 8, 30, 63]
        info = {'priority': [], 'replay_unique_id': [], 'replay_buffer_idx': []}
        for idx in selected_idx:
            info['priority'].append(np.random.uniform() + 64 - idx)
            info['replay_unique_id'].append(advanced_buffer._data[idx]['replay_unique_id'])
            info['replay_buffer_idx'].append(advanced_buffer._data[idx]['replay_buffer_idx'])

        for _ in range(8):
            advanced_buffer.push(generate_data(), 0)
        origin_data = copy.deepcopy(advanced_buffer._data)
        advanced_buffer.update(info)
        assert (np.argmax(info['priority']) == 0)
        assert (advanced_buffer._max_priority == max(info['priority'][2:]))
        assert (advanced_buffer._max_priority != max(info['priority']))
        for i in range(2):
            assert (origin_data[selected_idx[i]]['priority'] == advanced_buffer._data[selected_idx[i]]['priority'])
        eps = advanced_buffer._eps
        for i in range(2, 5):
            assert (info['priority'][i] + eps == advanced_buffer._data[selected_idx[i]]['priority'])
        # test case when data is None(such as max use remove)
        advanced_buffer._data[selected_idx[0]] = None
        advanced_buffer._valid_count -= 1
        advanced_buffer.update(info)

        # test beta
        advanced_buffer.beta = 1.
        assert (advanced_buffer.beta == 1.)

    def test_sample(self):
        buffer_cfg = deep_merge_dicts(
            AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=2))
        )
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        for _ in range(64):
            data = generate_data()
            data['priority'] = None
            advanced_buffer.push(data, 0)
        use_dict = defaultdict(int)
        while True:
            can_sample, _ = advanced_buffer._sample_check(32, 0)
            if not can_sample:
                break
            batch = advanced_buffer.sample(32, 0)
            assert (len(batch) == 32)
            assert (all([b['IS'] == 1.0 for b in batch])), [b['IS'] for b in batch]  # because priority is not updated
            idx = [b['replay_buffer_idx'] for b in batch]
            for i in idx:
                use_dict[i] += 1
        assert sum(map(lambda x: x[1] >= advanced_buffer._max_use,
                       use_dict.items())) == advanced_buffer.replay_buffer_size - advanced_buffer.count()
        for k, v in use_dict.items():
            if v > advanced_buffer._max_use:
                assert advanced_buffer._data[k] is None

        for _ in range(64):
            data = generate_data()
            data['priority'] = None
            advanced_buffer.push(data, 0)
        batch = advanced_buffer.sample(10, 0, sample_range=slice(-20, -2))
        assert len(batch) == 10

    def test_head_tail(self):
        buffer_cfg = deep_merge_dicts(
            AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=4))
        )
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        for i in range(65):
            advanced_buffer.push(generate_data(), 0)
        assert advanced_buffer._head == advanced_buffer._tail == 1
        info = {'replay_unique_id': [], 'replay_buffer_idx': [], 'priority': []}
        for data in advanced_buffer._data:
            info['replay_unique_id'].append(data['replay_unique_id'])
            info['replay_buffer_idx'].append(data['replay_buffer_idx'])
            info['priority'].append(0.)
        info['priority'][1] = 1000.
        advanced_buffer.update(info)
        while advanced_buffer._data[1] is not None:
            data = advanced_buffer.sample(1, 0)
            print(data)
        advanced_buffer.push({'data_id': '1096'}, 0)
        assert advanced_buffer._tail == 2
        assert advanced_buffer._head == 2

    def test_weight(self):
        buffer_cfg = deep_merge_dicts(
            AdvancedReplayBuffer.default_config(), EasyDict(dict(replay_buffer_size=64, max_use=1))
        )
        advanced_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')
        assert (advanced_buffer.count() == 0)  # assert empty buffer

        def get_weights(data_):
            weights_ = []
            for d in data_:
                if 'priority' not in d.keys() or d['priority'] is None:
                    weights_.append(advanced_buffer.max_priority)
                else:
                    weights_.append(d['priority'])
            weights_ = np.array(weights_)
            weights_ = weights_ ** advanced_buffer.alpha
            return weights_

        # first part(20 elements, smaller than buffer.replay_buffer_size)
        data = generate_data_list(20)
        advanced_buffer.push(data, 0)

        assert (advanced_buffer.replay_buffer_size == 64)
        assert (advanced_buffer.beta == 0.4)
        assert (advanced_buffer.alpha == 0.6)
        assert (hasattr(advanced_buffer, '_sum_tree'))
        assert (hasattr(advanced_buffer, '_min_tree'))
        assert (advanced_buffer.count() == 20)

        # tree test
        weights = get_weights(data)
        assert (np.fabs(weights.sum() - advanced_buffer._sum_tree.reduce()) < 1e-6)

        # second part(80 elements, bigger than buffer.replay_buffer_size)
        data = generate_data_list(80)
        advanced_buffer.push(data, 0)
        assert (advanced_buffer.count() == 64)
        assert (advanced_buffer._next_unique_id == 20 + 80)
        assert (advanced_buffer._tail == (20 + 80) % 64)
        weights = get_weights(data[-64:])
        assert (np.fabs(weights.sum() - advanced_buffer._sum_tree.reduce()) < 1e-6)
        weights = get_weights(data[-36:])
        assert (np.fabs(weights.sum() - advanced_buffer._sum_tree.reduce(start=0, end=36)) < 1e-6)

    @pytest.mark.rate
    def test_rate_limit(self):
        buffer_cfg = AdvancedReplayBuffer.default_config()
        buffer_cfg.replay_buffer_size = 1000
        buffer_cfg.thruput_controller = EasyDict(
            push_sample_rate_limit=dict(
                max=2,
                min=0.5,
            ),
            window_seconds=5,
            sample_min_limit_ratio=1.5,
        )
        prioritized_buffer = AdvancedReplayBuffer(buffer_cfg, tb_logger=None, instance_name='test')

        # Too many samples
        data = generate_data_list(30)
        prioritized_buffer.push(data, 0)  # push: 30
        for _ in range(3):
            _ = prioritized_buffer.sample(19, 0)  # sample: 3 * 19 = 57
        sampled_data = prioritized_buffer.sample(19, 0)
        assert sampled_data is None

        # Too big batch_size
        sampled_data = prioritized_buffer.sample(21, 0)
        assert sampled_data is None

        # Too many pushes
        assert prioritized_buffer.count() == 30
        for _ in range(2):
            data = generate_data_list(30)
            prioritized_buffer.push(data, 0)  # push: 30 + 2 * 30 = 90
        assert prioritized_buffer.count() == 90
        data = generate_data_list(30)
        prioritized_buffer.push(data, 0)
        assert prioritized_buffer.count() == 90

        # Test thruput_controller
        cur_sample_count = prioritized_buffer._thruput_controller.history_sample_count
        cur_push_count = prioritized_buffer._thruput_controller.history_push_count
        time.sleep(buffer_cfg.thruput_controller.window_seconds)
        assert abs(prioritized_buffer._thruput_controller.history_sample_count - cur_sample_count *
                   0.01) < 1e-5, (cur_sample_count, prioritized_buffer._thruput_controller.history_sample_count)
        assert abs(prioritized_buffer._thruput_controller.history_push_count - cur_push_count *
                   0.01) < 1e-5, (cur_push_count, prioritized_buffer._thruput_controller.history_push_count)


@pytest.mark.unittest(rerun=5)
class TestDemonstrationBuffer:

    def test_naive(self, setup_demo_buffer_factory):
        setup_demo_buffer = next(setup_demo_buffer_factory)
        naive_demo_buffer = next(setup_demo_buffer_factory)
        while True:
            with open(demo_data_path, 'rb+') as f:
                data = pickle.load(f)
            if len(data) != 0:
                break
            else:  # for the stability of dist-test
                demo_data = {'data': generate_data_list(10)}
                with open(demo_data_path, "wb") as f:
                    pickle.dump(demo_data, f)

        setup_demo_buffer.load_state_dict(data)
        assert setup_demo_buffer.count() == len(data['data'])  # assert buffer not empty
        samples = setup_demo_buffer.sample(3, 0)
        assert 'staleness' in samples[0]
        assert samples[1]['staleness'] == -1
        assert len(samples) == 3
        update_info = {'replay_unique_id': ['demo_0', 'demo_2'], 'replay_buffer_idx': [0, 2], 'priority': [1.33, 1.44]}
        setup_demo_buffer.update(update_info)
        samples = setup_demo_buffer.sample(10, 0)
        for sample in samples:
            if sample['replay_unique_id'] == 'demo_0':
                assert abs(sample['priority'] - 1.33) <= 0.01 + 1e-5, sample
            if sample['replay_unique_id'] == 'demo_2':
                assert abs(sample['priority'] - 1.44) <= 0.02 + 1e-5, sample

        state_dict = setup_demo_buffer.state_dict()
        naive_demo_buffer.load_state_dict(state_dict, deepcopy=True)
        assert naive_demo_buffer._tail == setup_demo_buffer._tail
        assert naive_demo_buffer._max_priority == setup_demo_buffer._max_priority

        os.popen('rm -rf log')
        os.popen('rm -rf {}'.format(demo_data_path))
