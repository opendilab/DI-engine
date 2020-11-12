import time
import threading
import queue
from typing import Iterable, Callable, Optional, Any
from collections import defaultdict

import torch
import torch.multiprocessing as multiprocessing
from nervex.torch_utils import to_device
from nervex.utils import LockContext, LockContextType
from .collate_fn import default_collate


class AsyncDataLoader(object):

    def __init__(
            self,
            data_source: Callable,
            batch_size: int,
            device: str,
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
        assert self.batch_size >= self.chunk_size and self.batch_size % self.chunk_size == 0, '{}/{}'.format(
            self.batch_size, self.chunk_size
        )
        self.num_workers = num_workers
        self.device = device
        self.use_cuda = 'cuda' in self.device
        if self.use_cuda:
            self.stream = torch.cuda.Stream()

        if self.num_workers < 0:
            raise ValueError(
                'num_workers option should be non-negative; '
                'use num_workers=0 or 1 to disable multiprocessing.'
            )
        queue_maxsize = max(1, self.num_workers) * 2
        self.queue_maxsize = queue_maxsize

        self.mp_context = multiprocessing.get_context('fork')
        self.manager = self.mp_context.Manager()
        self.async_train_queue = self.mp_context.Queue(maxsize=queue_maxsize)
        self.end_flag = False

        if self.num_workers > 1:
            self.batch_id = self.mp_context.Value('i', 0)
            self.cur_batch = self.mp_context.Value('i', 0)
            if self.batch_size != self.chunk_size:
                self.job_result = self.manager.dict()
                self.job_result_lock = LockContext(type_=LockContextType.PROCESS_LOCK)
            self.job_queue = self.mp_context.Queue(maxsize=queue_maxsize)
            self.worker = [self.mp_context.Process(target=self._worker_loop, args=()) for _ in range(self.num_workers)]
            for w in self.worker:
                w.daemon = True
                w.start()
            print('using {} workers loading data'.format(self.num_workers))

        self.async_process = self.mp_context.Process(target=self._async_loop, args=())
        self.async_process.daemon = True
        self.async_process.start()

        if self.use_cuda:
            self.cuda_queue = queue.Queue(maxsize=queue_maxsize)
            self.cuda_thread = threading.Thread(target=self._cuda_loop, args=())
            self.cuda_thread.daemon = True
            self.cuda_thread.start()

    def __iter__(self) -> Iterable:
        return self

    def _async_loop(self) -> None:
        while not self.end_flag:
            if self.num_workers > 1:
                if self.job_queue.full():
                    time.sleep(0.1)
                else:
                    data_fn = self.data_source(self.batch_size)
                    chunk_num = self.batch_size // self.chunk_size
                    with self.batch_id.get_lock():
                        for i in range(chunk_num):
                            start, end = i * self.chunk_size, (i + 1) * self.chunk_size
                            self.job_queue.put({'batch_id': self.batch_id.value, 'job': data_fn[start:end]})
                        self.batch_id.value = (self.batch_id.value + 1) % self.queue_maxsize
                    time.sleep(1.0)
            else:
                if self.async_train_queue.full():
                    time.sleep(0.1)
                else:
                    data_fn = self.data_source(self.batch_size)
                    data = [fn() for fn in data_fn]
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)

    def _worker_loop(self) -> None:
        while not self.end_flag:
            if self.job_queue.empty() or self.async_train_queue.full():
                time.sleep(0.1)
                continue
            else:
                element = self.job_queue.get()
                batch_id, job = element['batch_id'], element['job']
                data = [j() for j in job]
                if len(data) == self.batch_size == self.chunk_size:
                    while batch_id != self.cur_batch.value:
                        time.sleep(0.05)
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)
                    with self.cur_batch.get_lock():
                        self.cur_batch.value = (self.cur_batch.value + 1) % self.queue_maxsize
                else:
                    finish_flag = False
                    with self.job_result_lock:
                        if batch_id not in self.job_result:
                            self.job_result[batch_id] = data
                        elif len(self.job_result[batch_id]) + len(data) == self.batch_size:
                            data += self.job_result.pop(batch_id)
                            assert batch_id not in self.job_result
                            finish_flag = True
                        else:
                            self.job_result[batch_id] += data
                    if finish_flag:
                        data = self.collate_fn(data)
                        self.async_train_queue.put(data)
        while not self.job_queue.empty():
            _ = self.job_queue.get()
        self.job_queue.close()
        self.job_queue.join()

    def _cuda_loop(self) -> None:
        with torch.cuda.stream(self.stream):
            while not self.end_flag:
                if self.async_train_queue.empty() or self.cuda_queue.full():
                    time.sleep(0.1)
                else:
                    data = self.async_train_queue.get()
                    data = to_device(data, self.device)
                    self.cuda_queue.put(data)
        while not self.async_train_queue.empty():
            _ = self.async_train_queue.get()
        self.async_train_queue.close()
        self.async_train_queue.join()

    def __next__(self) -> Any:
        while not self.end_flag:
            if self.use_cuda:
                if self.cuda_queue.empty():
                    time.sleep(0.01)
                else:
                    return self.cuda_queue.get()
            else:
                if self.async_train_queue.empty():
                    time.sleep(0.01)
                else:
                    return self.async_train_queue.get()
        if self.use_cuda:
            while not self.cuda_queue.empty():
                _ = self.cuda_queue.get()
            self.cuda_queue.task_done()
            self.cuda_queue.join()
        else:
            while not self.async_train_queue.empty():
                _ = self.async_train_queue.get()
            self.async_train_queue.close()
            self.async_train_queue.join()

    def _clean_queue(self) -> None:
        while not self.async_train_queue.empty():  # pop all the data
            _ = self.async_train_queue.get()
        self.async_train_queue.close()
        self.async_train_queue.join_thread()  # let all the data in buffer written into pipe

        if self.use_cuda:
            while not self.cuda_queue.empty():
                _ = self.cuda_queue.get()
            self.cuda_queue.join()

    def __del__(self) -> None:
        self.end_flag = True
        self._clean_queue()
        self.async_process.terminate()
        self.async_process.join()
        if self.num_workers > 1:
            for w in self.worker:
                w.terminate()
                w.join()
        print('Del AsyncDataLoader')
