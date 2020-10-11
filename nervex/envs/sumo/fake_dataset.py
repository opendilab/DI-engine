import random

import torch


class FakeSumoDataset:

    def __init__(self):
        self.action_dim = [2, 2, 3]
        self.input_dim = 380

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

    def get_batch_sample(self, bs):
        batch = []
        for _ in range(bs):
            step = {}
            step['obs'] = self.get_random_obs()
            step['next_obs'] = self.get_random_obs()
            step['action'] = self.get_random_action()
            step['done'] = self.get_random_terminals()
            step['reward'] = self.get_random_reward()
            batch.append(step)
        return batch
