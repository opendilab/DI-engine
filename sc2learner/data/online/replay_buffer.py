from threading import Thread

from sc2learner.data.structure import PrioritizedBuffer, Cache
from sc2learner.utils import LockContext


class ReplayBuffer:
    """
    Overview: reinforcement learning replay buffer, with priority sampling, data cache
    Interface: __init__, push_data, sample, update, run, close
    """
    def __init__(self, cfg):
        """
        Overview: initialize replay buffer
        Arguments:
            - cfg (:obj:`dict`): config dict
        """
        self.cfg = cfg
        max_reuse = cfg.max_reuse if 'max_reuse' in cfg.keys() else None
        self._meta_buffer = PrioritizedBuffer(
            maxlen=cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=cfg.min_sample_ratio,
            alpha=cfg.alpha,
            beta=cfg.beta
        )
        # cache mechanism: first push data into cache, then(some conditions) put forward to meta buffer
        self._cache = Cache(maxlen=cfg.cache_maxlen, timeout=cfg.timeout)

        self._meta_lock = LockContext(lock_type='thread')
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

    def push_data(self, data):
        """
        Overview: push data into replay buffer
        Arguments:
            - data (:obj:`list` or `dict`): data list or data item
        Note: thread-safe
        """
        assert (isinstance(data, list) or isinstance(data, dict))

        if isinstance(data, list):
            for d in data:
                self._cache.push_data(d)
        elif isinstance(data, dict):
            self._cache.push_data(data)

    def sample(self, batch_size):
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

    def update(self, info):
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
