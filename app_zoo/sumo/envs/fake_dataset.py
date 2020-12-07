import random
import os

import torch
from nervex.utils import get_data_compressor


class FakeSumoDataset:

    def __init__(self, use_meta=False):
        self.action_dim = [2, 2, 3]
        self.input_dim = 380
        self.use_meta = use_meta
        self.output_dir = './data'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.count = 0
        self.compressor = get_data_compressor('lz4')

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
        return torch.FloatTensor([random.random() - 0.5])

    def get_random_terminals(self):
        sample = random.random()
        if sample > 0.99:
            return 1
        return 0

    def __getitem__(self, idx):
        step = {}
        step['obs'] = self.get_random_obs()
        step['next_obs'] = self.get_random_obs()
        step['action'] = self.get_random_action()
        step['done'] = self.get_random_terminals()
        step['reward'] = self.get_random_reward()
        if self.use_meta:
            path = os.path.join(self.output_dir, 'data_{}.pt'.format(self.count))
            data = self.compressor([step])
            torch.save(data, path)
            self.count += 1
            return {
                'job_id': self.count - 1,
                'traj_id': path,
                'priority': 1.0,
                'compressor': 'lz4',
                'data_push_length': 1
            }
        else:
            self.count += 1
            return step

    def get_batch_sample(self, bs):
        batch = []
        for _ in range(bs):
            batch.append(self.__getitem__(0))
        return batch
