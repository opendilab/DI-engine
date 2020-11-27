import time
import threading
import queue
from typing import Iterable, Callable, Optional, Any, Union
from collections import defaultdict

import torch
import torch.multiprocessing as tm
from nervex.torch_utils import to_device
from nervex.utils import LockContext, LockContextType
from .collate_fn import default_collate


class AsyncDataLoader(object):
    r"""
    Overview:
        An asynchronous dataloader.
    Interface:
        __init__, __iter__, __next__, __del__
    """

    def __init__(
            self,
            data_source: Union[Callable, dict],
            batch_size: int,
            device: str,
            chunk_size: Optional[int] = None,
            collate_fn: Optional[Callable] = None,
            num_workers: int = 0
    ) -> None:
        """
        Overview:
            Init dataloader with input parameters.
        Arguments:
            - data_source (:obj:`Union[Callable, dict]`): the data source, e.g. file, replay buffer...
            - batch_size (:obj:`int`): batch size
            - device (:obj:`str`): device
            - chunk_size (:obj:`int`): the size of chunked one in a batch, should exactly divide ``batch_size``, \
                only function when there are more than 1 worker.
            - collate_fn (:obj:`Callable`): the function used to collate batch size into each data field
            - num_workers (:obj:`int`): number of extra workers, implemented by using multiprocessing. \
                0 or 1 means only 1 main worker and no extra ones, in other words multiprocessing is disabled.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = 'cuda' in self.device
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        if chunk_size is None:
            self.chunk_size = 1
        else:
            self.chunk_size = chunk_size
        assert self.batch_size >= self.chunk_size and self.batch_size % self.chunk_size == 0, '{}/{}'.format(
            self.batch_size, self.chunk_size
        )
        if collate_fn is None:
            self.collate_fn = default_collate
        else:
            self.collate_fn = collate_fn
        self.num_workers = num_workers
        if self.num_workers < 0:
            raise ValueError(
                'num_workers should be non-negative; '
                'use num_workers = 0 or 1 to disable multiprocessing.'
            )
        queue_maxsize = max(1, self.num_workers) * 2
        self.queue_maxsize = queue_maxsize

        self.mp_context = tm.get_context('fork')
        self.manager = self.mp_context.Manager()
        # the queue to store processed data, user will get data from it if don't use cuda
        self.async_train_queue = self.mp_context.Queue(maxsize=queue_maxsize)
        self.end_flag = False

        # more than 1 worker to process data, use multiprocessing
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

        p, c = self.mp_context.Pipe()
        # async process (main worker): process data if num_workers <= 1; assign job to other workers if num_workers > 1
        self.async_process = self.mp_context.Process(target=self._async_loop, args=(p, c))
        self.async_process.daemon = True
        self.async_process.start()

        # cuda thread
        if self.use_cuda:
            # the queue to store processed cuda data, user will get data from it if use cuda
            self.cuda_queue = queue.Queue(maxsize=queue_maxsize)
            self.cuda_thread = threading.Thread(target=self._cuda_loop, args=())
            self.cuda_thread.daemon = True
            self.cuda_thread.start()

        # get data thread, coordinate with async process
        self.get_data_thread = threading.Thread(target=self._get_data, args=(p, c))
        self.get_data_thread.daemon = True
        self.get_data_thread.start()

    def __iter__(self) -> Iterable:
        """
        Overview:
            Return the iterable self as an iterator
        Returns:
            - self (:obj:`Iterable`): self as an iterator
        """
        return self

    def _get_data(self, p: tm.multiprocessing.connection, c: tm.multiprocessing.connection) -> None:
        """
        Overview:
            Init dataloader with input parameters. Will run as a thread through ``self.get_data_thread``.
        Arguments:
            - p (:obj:`tm.multiprocessing.connection`): parent connection
            - c (:obj:`tm.multiprocessing.connection`): child connection
        """
        c.close()  # close unused c, only use p
        while not self.end_flag:
            if not p.poll(timeout=0.2):
                time.sleep(0.01)
                continue
            try:
                cmd = p.recv()
            except EOFError:
                break
            if cmd == 'get_data':
                # main worker asks for data, should send it
                data = self.data_source(self.batch_size)
                # We expect ``data`` to be a function that should be implemented and processed,
                # therefore we can assign this job to all workers and complete it asynchronously;
                # But if we get a dict, which means the data has already been processed,
                # we can put it directly in async_train_queue and wait to be got by a user, e.g. learner.
                if isinstance(data[0], dict):
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)
                    p.send('pass')
                else:
                    p.send(data)
        p.close()

    def _async_loop(self, p: tm.multiprocessing.connection, c: tm.multiprocessing.connection) -> None:
        """
        Overview:
            Get data from ``self.get_data_thread``.
            If multiprocessing, put data in ``self.job_queue`` for further multiprocessing operation;
            If use only one worker, process data and put directly in ``self.async_train_queue``.
            Will run as a process through ``self.async_process``.
        Arguments:
            - p (:obj:`tm.multiprocessing.connection`): parent connection
            - c (:obj:`tm.multiprocessing.connection`): child connection
        """
        p.close()  # close unused p, only use c
        while not self.end_flag:
            if self.num_workers > 1:
                # multiprocessing, put jobs (chunked data) into job_queue
                if self.job_queue.full():
                    time.sleep(0.001)
                else:
                    c.send('get_data')
                    data = c.recv()
                    if isinstance(data, str) and data == 'pass':
                        continue
                    # chunk data into pieces and put it into job_queue
                    chunk_num = self.batch_size // self.chunk_size
                    with self.batch_id.get_lock():
                        for i in range(chunk_num):
                            start, end = i * self.chunk_size, (i + 1) * self.chunk_size
                            self.job_queue.put({'batch_id': self.batch_id.value, 'job': data[start:end]})
                        self.batch_id.value = (self.batch_id.value + 1) % self.queue_maxsize  # add batch_id
                    time.sleep(0.001)
            else:
                # only one worker, process data and directly put into async_train_queue
                if self.async_train_queue.full():
                    time.sleep(0.001)
                else:
                    c.send('get_data')
                    data = c.recv()
                    if isinstance(data, str) and data == 'pass':
                        continue
                    data = [fn() for fn in data]
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)
        c.close()

    def _worker_loop(self) -> None:
        """
        Overview:
            Get data from ``self.job_queue``, process it and then put into ``self.async_train_queue``.
            Only function when ``self.num_workers`` > 1, which means needs multiprocessing.
            Will run as a process through process elements in ``self.worker`` list.
        """
        while not self.end_flag:
            if self.job_queue.empty() or self.async_train_queue.full():
                # no left job or finished job cannot be stored
                time.sleep(0.01)
                continue
            else:
                element = self.job_queue.get()
                batch_id, job = element['batch_id'], element['job']
                data = [fn() for fn in job]  # only function-type job will arrive here, dict-type will not
                if len(data) == self.batch_size == self.chunk_size:
                    # data not chunked, finish the assigned one means finishing a whole batch
                    while batch_id != self.cur_batch.value:
                        # the job's batch is not current one, must wait for prior to be accomplished
                        time.sleep(0.01)
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)
                    # directly update cur_batch, since a whole batch is finished
                    with self.cur_batch.get_lock():
                        self.cur_batch.value = (self.cur_batch.value + 1) % self.queue_maxsize
                else:
                    # data chunked, must wait for all chunked pieces in a batch to be accomplished
                    finish_flag = False  # indicate whether a whole batch is accomplished
                    with self.job_result_lock:
                        if batch_id not in self.job_result:
                            # the first one in a batch
                            self.job_result[batch_id] = data
                        elif len(self.job_result[batch_id]) + len(data) == self.batch_size:
                            # the last one in a batch
                            data += self.job_result.pop(batch_id)
                            assert batch_id not in self.job_result
                            finish_flag = True
                        else:
                            # middle pieces in a batch
                            self.job_result[batch_id] += data
                    if finish_flag:
                        data = self.collate_fn(data)
                        while batch_id != self.cur_batch.value:
                            time.sleep(0.01)
                        self.async_train_queue.put(data)
                        with self.cur_batch.get_lock():
                            self.cur_batch.value = (self.cur_batch.value + 1) % self.queue_maxsize
        # self.end_flag is True, clear and close job_queue
        while not self.job_queue.empty():
            _ = self.job_queue.get()
        self.job_queue.close()
        self.job_queue.join()

    def _cuda_loop(self) -> None:
        """
        Overview:
            Only when using cuda, would this be run as a thread through ``self.cuda_thread``.
            Get data from ``self.async_train_queue``, change its device and put it into ``self.cuda_queue``
        """
        with torch.cuda.stream(self.stream):
            while not self.end_flag:
                if self.async_train_queue.empty() or self.cuda_queue.full():
                    time.sleep(0.01)
                else:
                    data = self.async_train_queue.get()
                    data = to_device(data, self.device)
                    self.cuda_queue.put(data)
        # self.end_flag is True, clear and close job_queue
        while not self.async_train_queue.empty():
            _ = self.async_train_queue.get()
        self.async_train_queue.close()
        self.async_train_queue.join()

    def __next__(self) -> Any:
        """
        Overview:
            Return next data in the iterator. If use cuda, get from ``self.cuda_queue``;
            Otherwise, get from ``self.async_train_queue``.
        Returns:
            - data (:obj:`torch.Tensor`): next data in the iterator
        """
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
        # self.end_flag is True, clear and close cuda_queue and async_train_queue
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
        """
        Overview:
            Clear and close async_train_queue, and cuda_queue if use cuda. Called by ``__del__``.
        """
        try:
            while not self.async_train_queue.empty():  # pop all the data
                _ = self.async_train_queue.get()
        except Exception:
            pass
        self.async_train_queue.close()
        self.async_train_queue.join()  # let all the data in buffer written into pipe
        if self.use_cuda:
            while not self.cuda_queue.empty():
                _ = self.cuda_queue.get()
            self.cuda_queue.join()

    def __del__(self) -> None:
        """
        Overview:
            Delete this dataloader. First clear and close all data queues, then terminate and join all processes.
        """
        if self.end_flag:
            return
        self.end_flag = True
        self._clean_queue()
        self.async_process.terminate()
        self.async_process.join()
        if self.num_workers > 1:
            for w in self.worker:
                w.terminate()
                w.join()
        print('Del AsyncDataLoader')
