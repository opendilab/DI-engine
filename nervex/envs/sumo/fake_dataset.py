import random
import tempfile
from collections import OrderedDict
import copy
import os
import numpy as np
import torch

from nervex.torch_utils import to_tensor
from nervex.utils import get_step_data_compressor


class FakeSumoDataset:
    def __init__(self, batch_size):
        self.action_dim = [2, 2, 3]
        self.input_dim = 380
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def get_random_action(self):
        action = []
        for i in self.action_dim:
            action.append(torch.randint(0, i - 1, size=(1, )))
        return action

    def get_random_obs(self):
        return torch.randn(380)

    def get_random_reward(self):
        return {'queue_len': torch.FloatTensor([random.random() - 0.5])}

    def get_random_terminals(self):
        sample = random.random()
        if sample > 0.99:
            return 1
        return 0

    def getBatchSample(self):
        while True:
            batch = []
            for _ in range(self.batch_size):
                step = {}
                step['obs'] = self.get_random_obs()
                step['next_obs'] = self.get_random_obs()
                step['action'] = self.get_random_action()
                step['done'] = self.get_random_terminals()
                step['reward'] = self.get_random_reward()
                batch.append(step)
            yield batch
