from typing import TYPE_CHECKING
from threading import Thread
from queue import Queue
import time
import torch
from easydict import EasyDict
from ding.data import Dataset, DataLoader
from ding.utils import get_rank
import numpy as np

if TYPE_CHECKING:
    from ding.framework import OfflineRLContext


class offline_data_fetcher_from_mem_c:

    def __init__(self, cfg: EasyDict, dataset: Dataset):
        stream = torch.cuda.Stream()
        def producer(queue, dataset, batch_size, device):
            torch.set_num_threads(4)
            nonlocal stream
            idx_iter = iter(np.random.permutation(len(dataset)))

            with torch.cuda.stream(stream):
                while True:
                    if queue.full():
                        time.sleep(0.1)
                    else:
                        try:
                            start_idx = next(idx_iter)
                        except StopIteration:
                            del idx_iter
                            idx_iter = iter(np.random.permutation(len(dataset)))
                            start_idx = next(idx_iter)
                        
                        data = [dataset.__getitem__(idx) for idx in range(start_idx, start_idx + batch_size)]
                        data = [[i[j] for i in data] for j in range(len(data[0]))]
                        try:
                            data = [torch.stack(x).to(device) for x in data]
                        except RuntimeError:
                            print(len(data))
                            for i in range(len(data)):
                                print(len(data[i]))
                                print(data[i])
                        queue.put(data)

        self.queue = Queue(maxsize=50)
        device = 'cuda:{}'.format(get_rank() % torch.cuda.device_count()) if cfg.policy.cuda else 'cpu'
        print('prepare sample data in device', device)
        self.producer_thread = Thread(
            target=producer,
            args=(self.queue, dataset, cfg.policy.batch_size, device),
            name='cuda_fetcher_producer'
        )

    def __call__(self,ctx: "OfflineRLContext"):
        if not self.producer_thread.is_alive():
            time.sleep(5)
            self.producer_thread.start()
        while self.queue.empty():
            time.sleep(0.001)
        ctx.train_data = self.queue.get()