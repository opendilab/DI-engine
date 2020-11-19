import copy
import os.path as osp
from threading import Thread
from typing import Union

from nervex.data.structure import PrioritizedBuffer, Cache
from nervex.utils import LockContext, LockContextType, read_config, deep_merge_dicts

default_config = read_config(osp.join(osp.dirname(__file__), 'replay_buffer_default_config.yaml')).replay_buffer


class ReplayBuffer:
    """
    Overview: reinforcement learning replay buffer, with priority sampling, data cache
    Interface: __init__, push_data, sample, update, run, close
    """

    def __init__(self, cfg: dict):
        """
        Overview: initialize replay buffer
        Arguments:
            - cfg (:obj:`dict`): config dict
        """
        self.cfg = deep_merge_dicts(default_config, cfg)
        max_reuse = self.cfg.max_reuse if 'max_reuse' in self.cfg.keys() else None
        self.traj_len = cfg.get('traj_len', None)
        self.unroll_len = cfg.get('unroll_len', None)
        self._meta_buffer = PrioritizedBuffer(
            maxlen=self.cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=self.cfg.min_sample_ratio,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            enable_track_used_data=self.cfg.enable_track_used_data
        )
        # cache mechanism: first push data into cache, then(some conditions) put forward to meta buffer
        self._cache = Cache(maxlen=self.cfg.cache_maxlen, timeout=self.cfg.timeout)

        self._meta_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        # from cache to meta data transport thread
        self._cache_thread = Thread(target=self._cache2meta)

    def _cache2meta(self):
        """
        Overview: get data from the cache and push it into meta buffer
        """
        # loop until the end flag is sent to the cache(the close method of the cache)
        for data in self._cache.get_cached_data_iter():
            with self._meta_lock:
                self._meta_buffer.append(data)

    def push_data(self, data: Union[list, dict]) -> None:
        """
        Overview: push data into replay buffer
        Arguments:
            - data (:obj:`list` or `dict`): data list or data item
        Note: thread-safe
        """
        assert (isinstance(data, list) or isinstance(data, dict))

        def push(item: dict) -> None:
            if 'data_push_length' not in item.keys():
                self._cache.push_data(item)
                return
            data_push_length = item['data_push_length']
            traj_len = self.traj_len if self.traj_len is not None else data_push_length
            unroll_len = self.unroll_len if self.unroll_len is not None else data_push_length
            assert data_push_length == traj_len
            split_num = traj_len // unroll_len
            split_item = [copy.deepcopy(item) for _ in range(split_num)]
            for i in range(split_num):
                split_item[i]['unroll_split_begin'] = i * unroll_len
                split_item[i]['unroll_len'] = unroll_len
                self._cache.push_data(split_item[i])

        if isinstance(data, list):
            for d in data:
                push(d)
        elif isinstance(data, dict):
            push(data)

    def sample(self, batch_size: int) -> list:
        """
        Overview: sample data from replay buffer
        Arguments:
            - batch_size (:obj:`int`): the batch size of the data will be sampled
        Returns:
            - data (:obj:`list` ): sampled data
        Note: thread-safe
        """
        with self._meta_lock:
            data = self._meta_buffer.sample(batch_size)
        return data

    def update(self, info: dict):
        """
        Overview: update meta buffer with outside info
        Arguments:
            - info (:obj:`dict`): info dict
        Note: thread-safe
        """
        with self._meta_lock:
            self._meta_buffer.update(info)

    def run(self):
        """
        Overview: launch the cache and cache2meta thread
        """
        self._cache.run()
        self._cache_thread.start()

    def close(self):
        """
        Overview: shut down the cache gracefully
        """
        self._cache.close()

    @property
    def count(self):
        """
        Overview: return current buffer data count
        """
        return self._meta_buffer.validlen

    @property
    def used_data(self):
        """
        Overview: return the used data(which is thrown from the buffer)
        """
        return self._meta_buffer.used_data
