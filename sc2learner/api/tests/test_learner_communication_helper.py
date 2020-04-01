import pytest
import time
import os
import sys
import torch
import numpy as np


def generate_data():
    return {'raw_data': torch.randn(4)}


def train(data):
    time.sleep(3+np.random.randint(-1, 2))
    info = {'replay_buffer_idx': [], 'replay_unique_id': [], 'priority': []}
    for d in data:
        info['replay_buffer_idx'].append(d['replay_buffer_idx'])
        info['replay_unique_id'].append(d['replay_unique_id'])
        info['priority'].append(d['priority'] + np.random.uniform(0, 1))
    return info


class TestLearnerCommHelper:

    def test_data_sample_update(self, coordinator, learner):
        """
        Note: coordinator must be in the front of learner in the arguments
        """
        for i in range(1024):
            coordinator.replay_buffer.push_data(generate_data())
        #assert (1024 == coordinator.replay_buffer._meta_buffer.validlen)

        time.sleep(1)

        for i in range(10):
            print('-'*20+'Training Iteration {}'.format(i)+'-'*20)
            print(coordinator.replay_buffer._meta_buffer.validlen, id(coordinator.replay_buffer._meta_buffer))
            # print(coordinator.replay_buffer._meta_buffer.sum_tree.reduce())
            if i % 2 == 0:
                data = coordinator.replay_buffer.sample(128)
            else:
                data = learner.sample()
            assert isinstance(data, list)
            info = train(data)
