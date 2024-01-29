from typing import TYPE_CHECKING
from threading import Thread, Event
from queue import Queue
import time
import numpy as np
import torch
from easydict import EasyDict
from ding.framework import task
from ding.data import Dataset, DataLoader
from ding.utils import get_rank, get_world_size

if TYPE_CHECKING:
    from ding.framework import OfflineRLContext


class OfflineMemoryDataFetcher:

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.FETCHER):
            return task.void()
        return super(OfflineMemoryDataFetcher, cls).__new__(cls)

    def __init__(self, cfg: EasyDict, dataset: Dataset):
        device = 'cuda:{}'.format(get_rank() % torch.cuda.device_count()) if cfg.policy.cuda else 'cpu'
        if device != 'cpu':
            stream = torch.cuda.Stream()

        def producer(queue, dataset, batch_size, device, event):
            torch.set_num_threads(4)
            if device != 'cpu':
                nonlocal stream
            sbatch_size = batch_size * get_world_size()
            rank = get_rank()
            idx_list = np.random.permutation(len(dataset))
            temp_idx_list = []
            for i in range(len(dataset) // sbatch_size):
                temp_idx_list.extend(idx_list[i + rank * batch_size:i + (rank + 1) * batch_size])
            idx_iter = iter(temp_idx_list)

            if device != 'cpu':
                with torch.cuda.stream(stream):
                    while True:
                        if queue.full():
                            time.sleep(0.1)
                        else:
                            data = []
                            for _ in range(batch_size):
                                try:
                                    data.append(dataset.__getitem__(next(idx_iter)))
                                except StopIteration:
                                    del idx_iter
                                    idx_list = np.random.permutation(len(dataset))
                                    idx_iter = iter(idx_list)
                                    data.append(dataset.__getitem__(next(idx_iter)))
                            data = [[i[j] for i in data] for j in range(len(data[0]))]
                            data = [torch.stack(x).to(device) for x in data]
                            queue.put(data)
                        if event.is_set():
                            break
            else:
                while True:
                    if queue.full():
                        time.sleep(0.1)
                    else:
                        data = []
                        for _ in range(batch_size):
                            try:
                                data.append(dataset.__getitem__(next(idx_iter)))
                            except StopIteration:
                                del idx_iter
                                idx_list = np.random.permutation(len(dataset))
                                idx_iter = iter(idx_list)
                                data.append(dataset.__getitem__(next(idx_iter)))
                        data = [[i[j] for i in data] for j in range(len(data[0]))]
                        data = [torch.stack(x) for x in data]
                        queue.put(data)
                    if event.is_set():
                        break

        self.queue = Queue(maxsize=50)
        self.event = Event()
        self.producer_thread = Thread(
            target=producer,
            args=(self.queue, dataset, cfg.policy.batch_size, device, self.event),
            name='cuda_fetcher_producer'
        )

    def __call__(self, ctx: "OfflineRLContext"):
        if not self.producer_thread.is_alive():
            time.sleep(5)
            self.producer_thread.start()
        while self.queue.empty():
            time.sleep(0.001)
        ctx.train_data = self.queue.get()

    def __del__(self):
        if self.producer_thread.is_alive():
            self.event.set()
            del self.queue
