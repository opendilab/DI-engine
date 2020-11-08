import time
from typing import Iterable, Callable, Optional, Any
from collections import defaultdict

import torch
import torch.multiprocessing as multiprocessing
from nervex.utils import LockContext, LockContextType
from .collate_fn import default_collate


class AsyncDataLoader(object):

    def __init__(
            self,
            data_source: Callable,
            batch_size: int,
            chunk_size: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
            num_workers: int = 0
    ) -> None:
        self.data_source = data_source
        if collate_fn is None:
            self.collate_fn = default_collate
        else:
            self.collate_fn = collate_fn
        self.batch_size = batch_size
        if chunk_size is None:
            self.chunk_size = 1
        else:
            self.chunk_size = chunk_size
        assert self.batch_size % self.chunk_size == 0, '{}/{}'.format(self.batch_size, self.chunk_size)
        self.num_workers = num_workers

        if self.num_workers < 0:
            raise ValueError(
                'num_workers option should be non-negative; '
                'use num_workers=0 or 1 to disable multiprocessing.'
            )

        self.async_train_queue = multiprocessing.Queue(maxsize=self.num_workers * 2)
        self.end_flag = False

        if self.num_workers > 1:
            self.batch_id = 0
            self.job_result = multiprocessing.Manager().dict()
            self.job_result_lock = LockContext(type_=LockContextType.PROCESS_LOCK)
            self.job_queue = multiprocessing.Queue(maxsize=self.num_workers * 2)
            self.worker = [multiprocessing.Process(target=self._worker_loop, args=()) for _ in range(self.num_workers)]
            for w in self.worker:
                w.daemon = True
                w.start()
            print('using {} workers loading data'.format(self.num_workers))

        self.async_process = multiprocessing.Process(target=self._async_loop, args=())
        self.async_process.daemon = True
        self.async_process.start()

    def __iter__(self) -> Iterable:
        return self

    def _async_loop(self) -> None:
        while not self.end_flag:
            if self.async_train_queue.full():
                time.sleep(0.1)
            else:
                data_fn = self.data_source(self.batch_size)
                if self.num_workers > 1:
                    chunk_num = self.batch_size // self.chunk_size
                    for i in range(chunk_num):
                        start, end = i * self.chunk_size, (i + 1) * self.chunk_size
                        self.job_queue.put({'batch_id': self.batch_id, 'job': data_fn[start:end]})
                    self.batch_id = (self.batch_id + 1) % (self.job_queue._maxsize * 2)
                    time.sleep(1.0)
                else:
                    data = [fn() for fn in data_fn]
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)

    def __next__(self) -> Any:
        while True:
            if self.async_train_queue.empty():
                time.sleep(0.001)
            else:
                return self.async_train_queue.get()

    def _worker_loop(self) -> None:
        while not self.end_flag:
            if self.async_train_queue.full() or self.job_queue.empty():
                time.sleep(0.1)
                continue
            else:
                element = self.job_queue.get()
                batch_id, job = element['batch_id'], element['job']
                data = [j() for j in job]
                with self.job_result_lock:
                    if batch_id not in self.job_result:
                        self.job_result[batch_id] = data
                    else:
                        self.job_result[batch_id] += data
                    if len(self.job_result[batch_id]) == self.batch_size:
                        data = self.job_result.pop(batch_id)
                        assert batch_id not in self.job_result
                        data = self.collate_fn(data)
                        self.async_train_queue.put(data)

    def __del__(self) -> None:
        self.end_flag = True
        self.async_process.terminate()
        self.async_process.join()
        if self.num_workers > 1:
            for w in self.worker:
                w.terminate()
                w.join()
        print('Del AsyncDataLoader')
