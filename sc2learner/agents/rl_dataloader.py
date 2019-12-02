from collections import deque
from torch.utils.data import _utils
import random


class RLBaseDataset(object):
    def __init__(self, maxlen, transform):
        self.queue = deque(maxlen)
        self.transform = transform

    def push(self):
        pass

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        return self.transform(self.queue[idx])


class RLBaseDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        if collate_fn is None:
            self.collate_fn = _utils.collate.default_collate
        self.batch_size = batch_size

    def _get_indices(self):
        indices = random.sample([i for i in range(len(self.dataset))], self.batch_size)
        return indices

    def __next__(self):
        indices = self._get_indices()
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
