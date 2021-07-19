from typing import Iterable, Callable, Optional, Any, Union
import time
import platform
import threading
import queue

import torch
import torch.multiprocessing as tm
from ding.torch_utils import to_device
from ding.utils import LockContext, LockContextType
from .base_dataloader import IDataLoader
from .collate_fn import default_collate


class AsyncDataLoader(IDataLoader):
    r"""
    Overview:
        An asynchronous dataloader.
    Interface:
        __init__, __iter__, __next__, close
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
            If ``data_source`` is ``dict``, data will only be processed in ``get_data_thread`` and put into
            ``async_train_queue``.
            If ``data_source`` is ``Callable``, data will be processed by implementing functions, and can be sorted
            in two types:

                - ``num_workers`` == 0 or 1: Only main worker will process it and put into ``async_train_queue``.
                - ``num_workers`` >  1: Main worker will divide a job into several pieces, push every job into \
                    ``job_queue``; Then slave workers get jobs and implement; Finally they will push procesed data \
                    into ``async_train_queue``.

            At the last step, if ``device`` contains "cuda", data in ``async_train_queue`` will be transferred to
            ``cuda_queue`` for uer to access.
        Arguments:
            - data_source (:obj:`Union[Callable, dict]`): The data source, e.g. function to be implemented(Callable), \
                replay buffer's real data(dict), etc.
            - batch_size (:obj:`int`): Batch size.
            - device (:obj:`str`): Device.
            - chunk_size (:obj:`int`): The size of a chunked piece in a batch, should exactly divide ``batch_size``, \
                only function when there are more than 1 worker.
            - collate_fn (:obj:`Callable`): The function which is used to collate batch size into each data field.
            - num_workers (:obj:`int`): Number of extra workers. \
                0 or 1 means only 1 main worker and no extra ones, i.e. Multiprocessing is disabled. \
                More than 1 means multiple workers implemented by multiprocessing are to processs data respectively.
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
                '"num_workers" should be non-negative; '
                'Use num_workers = 0 or 1 to disable multiprocessing.'
            )
        # Up to "2 * num_workers" pieces of data will be stored in dataloader, waiting for learner to get.
        # Up to "2 * num_workers" jobs will be stored in dataloader, waiting for slave process to get and accomplish.
        queue_maxsize = max(1, self.num_workers) * 2
        self.queue_maxsize = queue_maxsize

        # For multiprocessing: Use ``spawn`` on Windows, ``fork`` on other platforms.
        context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        self.mp_context = tm.get_context(context_str)
        self.manager = self.mp_context.Manager()
        # ``async_train_queue`` is the queue to store processed data.
        # User can directly access data if don't use cuda; Otherwise, user will access data from ``cuda_queue``.
        self.async_train_queue = self.mp_context.Queue(maxsize=queue_maxsize)
        self.end_flag = False

        # Multiprocessing workers: If num_workers > 1, more than 1 worker are to process data.
        if self.num_workers > 1:
            self.batch_id = self.mp_context.Value('i', 0)
            self.cur_batch = self.mp_context.Value('i', 0)
            if self.batch_size != self.chunk_size:
                # job_result {batch_id: result_list} is used to store processed result in temporal.
                self.job_result = self.manager.dict()
                self.job_result_lock = LockContext(type_=LockContextType.PROCESS_LOCK)
            self.job_queue = self.mp_context.Queue(maxsize=queue_maxsize)
            self.worker = [
                self.mp_context.Process(
                    target=self._worker_loop, args=(), name='dataloader_worker{}_{}'.format(i, time.time())
                ) for i in range(self.num_workers)
            ]
            for w in self.worker:
                w.daemon = True
                w.start()
            print('Using {} workers to load data'.format(self.num_workers))

        # Parent and child pipes. Used by ``async_process`` and ``get_data_thread`` to coordinate.
        p, c = self.mp_context.Pipe()

        # Async process (Main worker): Process data if num_workers <= 1; Assign job to other workers if num_workers > 1.
        self.async_process = self.mp_context.Process(target=self._async_loop, args=(p, c))
        self.async_process.daemon = True
        self.async_process.start()

        # Get data thread: Get data from ``data_source`` and send it to ``async_process``.`
        self.get_data_thread = threading.Thread(target=self._get_data, args=(p, c))
        self.get_data_thread.daemon = True
        self.get_data_thread.start()

        # Cuda thread: If use cuda, data in ``async_train_queue`` will be transferred to ``cuda_queue``;
        # Then user will access data from ``cuda_queue``.
        if self.use_cuda:
            self.cuda_queue = queue.Queue(maxsize=queue_maxsize)
            self.cuda_thread = threading.Thread(target=self._cuda_loop, args=(), name='dataloader_cuda')
            self.cuda_thread.daemon = True
            self.cuda_thread.start()

    def __iter__(self) -> Iterable:
        """
        Overview:
            Return the iterable self as an iterator.
        Returns:
            - self (:obj:`Iterable`): Self as an iterator.
        """
        return self

    def _get_data(self, p: tm.multiprocessing.connection, c: tm.multiprocessing.connection) -> None:
        """
        Overview:
            Init dataloader with input parameters. Will run as a thread through ``self.get_data_thread``.
        Arguments:
            - p (:obj:`tm.multiprocessing.connection`): Parent connection.
            - c (:obj:`tm.multiprocessing.connection`): Child connection.
        """
        c.close()  # Close unused c, only use p
        while not self.end_flag:
            if not p.poll(timeout=0.2):
                time.sleep(0.01)
                continue
            try:
                cmd = p.recv()
            except EOFError:
                break
            if cmd == 'get_data':
                # Main worker asks for data.
                data = self.data_source(self.batch_size)
                # ``data`` can be callable, e.g. a function to read data from file, therefore we can divide
                # this job to pieces, assign to every slave worker and accomplish jobs asynchronously.
                # But if we get a list of dicts, which means the data has already been processed and
                # can be used directly, we can put it directly in async_train_queue and wait it
                # to be accessed by a user, e.g. learner.
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
            Main worker process. Run through ``self.async_process``.
            Firstly, get data from ``self.get_data_thread``.
            If multiple workers, put data in ``self.job_queue`` for further multiprocessing operation;
            If only one worker, process data and put directly into ``self.async_train_queue``.
        Arguments:
            - p (:obj:`tm.multiprocessing.connection`): Parent connection.
            - c (:obj:`tm.multiprocessing.connection`): Child connection.
        """
        p.close()  # Close unused p, only use c
        while not self.end_flag:
            if self.num_workers > 1:
                # Multiple workers: Put jobs (chunked data) into job_queue
                if self.job_queue.full():
                    time.sleep(0.001)
                else:
                    # Get data from ``_get_data`` thread.
                    c.send('get_data')
                    data = c.recv()
                    if isinstance(data, str) and data == 'pass':
                        continue
                    # Get data to be processed, chunk it into pieces and put them into job_queue.
                    chunk_num = self.batch_size // self.chunk_size
                    with self.batch_id.get_lock():
                        for i in range(chunk_num):
                            start, end = i * self.chunk_size, (i + 1) * self.chunk_size
                            self.job_queue.put({'batch_id': self.batch_id.value, 'job': data[start:end]})
                        self.batch_id.value = (self.batch_id.value + 1) % self.queue_maxsize  # Increment batch_id
                    time.sleep(0.001)
            else:
                # Only one worker: Process data and directly put it into async_train_queue
                if self.async_train_queue.full():
                    time.sleep(0.001)
                else:
                    c.send('get_data')
                    data = c.recv()
                    if isinstance(data, str) and data == 'pass':
                        continue
                    data = [fn() for fn in data]  # Implement functions in list ``data``.
                    data = self.collate_fn(data)
                    self.async_train_queue.put(data)
        c.close()

    def _worker_loop(self) -> None:
        """
        Overview:
            Worker process. Run through each element in list ``self.worker``.
            Get data job from ``self.job_queue``, process it and then put into ``self.async_train_queue``.
            Only function when ``self.num_workers`` > 1, which means using multiprocessing.
        """
        while not self.end_flag:
            if self.job_queue.empty() or self.async_train_queue.full():
                # No left job to be done, or finished job have no space to store.
                time.sleep(0.01)
                continue
            else:
                try:
                    element = self.job_queue.get()
                except (ConnectionResetError, ConnectionRefusedError) as e:
                    break
                batch_id, job = element['batch_id'], element['job']
                # Process the assigned data.
                data = [fn() for fn in job]  # Only function-type job will arrive here, dict-type will not
                if len(data) == self.batch_size == self.chunk_size:
                    # Data not chunked: Finish the assigned one means finishing a whole batch.
                    data = self.collate_fn(data)
                    while batch_id != self.cur_batch.value:
                        time.sleep(0.01)
                    self.async_train_queue.put(data)
                    # Directly update cur_batch, since a whole batch is finished
                    with self.cur_batch.get_lock():
                        self.cur_batch.value = (self.cur_batch.value + 1) % self.queue_maxsize
                else:
                    # Data chunked: Must wait for all chunked pieces in a batch to be accomplished.
                    finish_flag = False  # indicate whether a whole batch is accomplished
                    with self.job_result_lock:
                        if batch_id not in self.job_result:
                            # The first one in a batch
                            self.job_result[batch_id] = data
                        elif len(self.job_result[batch_id]) + len(data) == self.batch_size:
                            # The last one in a batch
                            data += self.job_result.pop(batch_id)
                            assert batch_id not in self.job_result
                            finish_flag = True
                        else:
                            # Middle pieces in a batch
                            self.job_result[batch_id] += data
                    if finish_flag:
                        data = self.collate_fn(data)
                        while batch_id != self.cur_batch.value:
                            time.sleep(0.01)
                        self.async_train_queue.put(data)
                        with self.cur_batch.get_lock():
                            self.cur_batch.value = (self.cur_batch.value + 1) % self.queue_maxsize
        # If ``self.end_flag`` is True, clear and close job_queue, because _worker_loop gets jobs from job_queue.
        while not self.job_queue.empty():
            try:
                _ = self.job_queue.get()
            except Exception as e:
                break
        self.job_queue.close()
        self.job_queue.join_thread()

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
        # If ``self.end_flag``` is True, clear and close async_train_queue,
        # because _cuda_loop gets data from async_train_queue.
        while not self.async_train_queue.empty():
            _ = self.async_train_queue.get()
        self.async_train_queue.close()
        self.async_train_queue.join_thread()

    def __next__(self) -> Any:
        """
        Overview:
            Return next data in the iterator. If use cuda, get from ``self.cuda_queue``;
            Otherwise, get from ``self.async_train_queue``.
        Returns:
            - data (:obj:`torch.Tensor`): Next data in the dataloader iterator.
        """
        while not self.end_flag:
            if self.use_cuda:
                if self.cuda_queue.empty():
                    time.sleep(0.01)
                else:
                    data = self.cuda_queue.get(timeout=60)
                    self.cuda_queue.task_done()
                    return data
            else:
                if self.async_train_queue.empty():
                    time.sleep(0.01)
                else:
                    return self.async_train_queue.get()
        # If ``self.end_flag``` is True, clear and close either 1) or 2):
        # 1) cuda_queue. Because user get data from cuda_queue, and async_train_queue is closed by cuda_loop.
        # 2) async_train_queue. Because user get data from async_train_queue.
        if self.use_cuda:
            while not self.cuda_queue.empty():
                _ = self.cuda_queue.get()
                self.cuda_queue.task_done()
            self.cuda_queue.join()
        else:
            while not self.async_train_queue.empty():
                _ = self.async_train_queue.get()
            self.async_train_queue.close()
            self.async_train_queue.join_thread()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """
        Overview:
            Delete this dataloader. First set ``end_flag`` to True, which means different processes/threads
            will clear and close all data queues; Then  all processes will be terminated and joined.
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.async_process.terminate()
        self.async_process.join()
        if self.num_workers > 1:
            for w in self.worker:
                w.terminate()
                w.join()
        print('Del AsyncDataLoader')
