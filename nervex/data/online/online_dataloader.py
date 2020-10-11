import torch
from collections.abc import Iterator
from torch.utils.data import _utils
import torch.multiprocessing as multiprocessing
from torch._six import queue
import time
from nervex.utils import LockContext


class OnlineDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        if collate_fn is None:
            self.collate_fn = _utils.collate.default_collate
        else:
            self.collate_fn = collate_fn
        self.batch_size = batch_size

    def __next__(self):
        batch, avg_usage, push_count, avg_model_index = \
            self.dataset.get_sample_batch(
                self.batch_size, self.cur_model_index)
        batch = self.collate_fn(batch)
        return batch, avg_usage, push_count, avg_model_index

    @property
    def cur_model_index(self):
        return self._cur_model_index

    @cur_model_index.setter
    def cur_model_index(self, cur_model_index):
        self._cur_model_index = cur_model_index


class OnlineIteratorDataLoader:
    def __init__(self, data_iterator, batch_size, collate_fn=None, read_data_fn=None, num_workers=0):
        assert (isinstance(data_iterator, Iterator))
        assert (read_data_fn is not None)
        self.data_iterator = data_iterator
        if collate_fn is None:
            self.collate_fn = _utils.collate.default_collate
        else:
            self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.read_data_fn = read_data_fn
        self.num_workers = num_workers

        if self.num_workers < 0:
            raise ValueError(
                'num_workers option should be non-negative; '
                'use num_workers=0 to disable multiprocessing.'
            )

        self.lock = LockContext(lock_type='process')
        if self.num_workers > 0:
            self.shared_index = torch.tensor(0)
            self.shared_index.share_memory_()
            self.put_index = torch.tensor(0)
            self.put_index.share_memory_()
            self.data_queue = multiprocessing.Queue()
            self.max_length = 10 * self.num_workers
            for i in range(self.num_workers):
                p = multiprocessing.Process(target=self._worker_loop, args=(i,))
                p.start()
            print('using {} workers loading data'.format(self.num_workers))

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_workers == 0:
            # TODO(nyz) why this line code must be wrappered by lock
            with self.lock:
                data = next(self.data_iterator)
            data = [self.read_data_fn(d) for d in data]
        else:
            while True:
                if self.data_queue.qsize() > 0:
                    data = self.data_queue.get()
                    break
                else:
                    print('waiting for loading data ...')
                    time.sleep(1)
        data = self.collate_fn(data)
        return data

    def _worker_loop(self, thread_id):
        while True:
            if self.data_queue.qsize() < self.max_length:
                with self.lock:
                    index = int(self.shared_index.item())
                    self.shared_index += 1
                    data = next(self.data_iterator)
                data = [self.read_data_fn(d) for d in data]
                while True:
                    if index - self.put_index == 1:
                        self.data_queue.put(data)
                        with self.lock:
                            self.put_index += 1
                        break
                    time.sleep(0.1)
            time.sleep(0.1)


def unroll_split_collate_fn(*args, collate_fn=_utils.collate.default_collate, **kwargs):
    # TODO: replace this hacky workaround for non unique sized data chunks
    # result = collate_fn(*args, **kwargs)
    # Expecting a list of dict as input
    # there are multiple samples in each key of dict (as a list or Tensor)
    # Returning a single dict with all samples in each key (as Tensor if possible)
    result = args[0]
    new_result = {}
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        for k, v in item.items():
            if isinstance(v, list) and v[0] == 'none':
                new_result[k] = None
            elif isinstance(v, str) and v == 'none':
                new_result[k] = None
            elif isinstance(v, torch.Tensor):
                if k in new_result:
                    new_result[k] = torch.cat((new_result[k], v))
                else:
                    new_result[k] = v
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                if k in new_result:
                    new_result[k] = torch.cat((new_result[k], torch.cat(v)))
                else:
                    new_result[k] = torch.cat(v)
            elif isinstance(v, list):
                if k in new_result:
                    new_result[k].extend(v)
                else:
                    new_result[k] = v
            else:
                print('WARNING: item {} of type {} in data discarded'.format(k, str(type(v))))
    return new_result
