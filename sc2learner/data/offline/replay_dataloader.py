import random
from .replay_dataset import START_STEP
from collections import deque
from sc2learner.utils import get_rank, get_world_size


class ReplayIterationDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        assert collate_fn is not None
        self.batch_size = batch_size
        self.rank = get_rank()
        self.world_size = get_world_size()

        L = len(self.dataset)
        indices = list(range(L))
        total_size = L // self.world_size * self.world_size  # discard the last data
        self.size = total_size // self.world_size
        indices = indices[:total_size]

        # TODO(pzh) At risk!! Is this seeding really effective?
        random.seed(1e9 + 314)  # fix seed for all the process

        random.shuffle(indices)
        data_index = indices[self.rank * self.size:(self.rank + 1) * self.size]

        self.data_index = deque(data_index)
        self.cur_data_index = []
        for _ in range(self.batch_size):
            self.cur_data_index.append(self.data_index.popleft())

    def __iter__(self):
        return self

    def __next__(self):
        end_list = self.dataset.step(self.cur_data_index)
        batch = [self.dataset[i] for i in self.cur_data_index]
        batch = self.collate_fn(batch)
        # update replay
        if len(end_list) > 0:
            self.dataset.reset_step(end_list)
            for end_index in end_list:
                idx = self.cur_data_index.index(end_index)
                tmp = self.cur_data_index[idx]
                self.cur_data_index[idx] = self.data_index.popleft()
                self.data_index.append(tmp)
        assert START_STEP in batch[0]
        return batch
