from typing import TYPE_CHECKING
from threading import Thread, Event
from queue import Queue
import time
import torch
import torch.distributed as dist
from easydict import EasyDict
from ding.framework import task
from ding.data import Dataset, DataLoader
from ding.utils import get_rank
import numpy as np

if TYPE_CHECKING:
    from ding.framework import OfflineRLContext


class offline_data_fetcher_from_mem_c:

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not task.has_role(task.role.FETCHER):
            return task.void()
        return super(offline_data_fetcher_from_mem_c, cls).__new__(cls)

    def __init__(self, cfg: EasyDict, dataset: Dataset):
        stream = torch.cuda.Stream()

        def producer(queue, dataset, batch_size, device, event):
            torch.set_num_threads(4)
            nonlocal stream
            num_gpu = dist.get_world_size()
            rank = get_rank()
            idx_list = np.random.permutation(len(dataset))
            temp_idx_list = []
            for i in range(len(dataset) // (batch_size * num_gpu)):
                temp_idx_list.extend(idx_list[i + rank * batch_size:i + (rank + 1) * batch_size])
            idx_iter = iter(temp_idx_list)

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

        self.queue = Queue(maxsize=50)
        self.event = Event()
        device = 'cuda:{}'.format(get_rank() % torch.cuda.device_count()) if cfg.policy.cuda else 'cpu'
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
