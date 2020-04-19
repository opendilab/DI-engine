import pytest
import time
import os
import sys
import threading
from threading import Thread
import torch
import numpy as np
from sc2learner.data import FakeActorDataset


def train(data):
    time.sleep(2 + np.random.randint(-1, 2))
    info = {'replay_buffer_idx': [], 'replay_unique_id': [], 'priority': []}
    for d in data:
        info['replay_buffer_idx'].append(d['replay_buffer_idx'])
        info['replay_unique_id'].append(d['replay_unique_id'])
        info['priority'].append(d['priority'] + np.random.uniform(0, 1))
    return info


@pytest.mark.unittest
class TestLearnerCommHelper:
    def fake_push_data(self, coordinator):
        time.sleep(3)  # monitor empty replay_buffer state
        dataset = FakeActorDataset(use_meta=True)
        for i in range(1024):
            coordinator.replay_buffer.push_data(dataset[i])
        time.sleep(1)  # wait the cache flush out
        assert (1024 == coordinator.replay_buffer._meta_buffer.validlen)

    def test_data_sample_update(self, coordinator, learner):
        """
        Note: coordinator must be in the front of learner in the arguments
        """
        push_data_thread = Thread(target=self.fake_push_data, args=(coordinator, ))
        push_data_thread.daemon = True
        push_data_thread.start()
        handle = coordinator.replay_buffer._meta_buffer

        for i in range(10):
            print('-' * 20 + 'Training Iteration {}'.format(i) + '-' * 20)
            print('current replay_buffer len: {}'.format(handle.validlen))
            print('current replay_buffer priority sum: {}'.format(handle.sum_tree.reduce()))
            data = next(learner.data_iterator)
            assert isinstance(data, list)
            assert len(data) == learner.batch_size
            info = train(data)
            assert learner.update_info(info)
            assert learner.register_model_in_coordinator('test_model')
            assert len(coordinator.learner_record[learner.learner_uid]['models']) == i + 1

        push_data_thread.join()
