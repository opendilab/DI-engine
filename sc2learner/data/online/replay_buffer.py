from threading import Thread

from sc2learner.data.structure import PrioritizedBuffer, Cache
from sc2learner.utils import LockContext


class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        max_reuse = cfg.max_reuse if 'max_reuse' in cfg.keys() else None
        self._meta_buffer = PrioritizedBuffer(
            maxlen=cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=cfg.min_sample_ratio,
            alpha=cfg.alpha,
            beta=cfg.beta
        )
        self._cache = Cache(maxlen=cfg.cache_maxlen, timeout=cfg.timeout)

        self._meta_lock = LockContext(lock_type='thread')
        self._cache_thread = Thread(target=self._cache2meta)

    def _cache2meta(self):
        for data in self._cache.get_cached_data_iter():
            with self._meta_lock:
                self._meta_buffer.append(data)

    def push_data(self, data):
        assert (isinstance(data, list) or isinstance(data, dict))

        if isinstance(data, list):
            for d in data:
                self._cache.push_data(d)
        elif isinstance(data, dict):
            self._cache.push_data(data)

    def sample(self, batch_size):
        with self._meta_lock:
            data = self._meta_buffer.sample(batch_size)
        return data

    def update(self, info):
        with self._meta_lock:
            self._meta_buffer.update(info)

    def run(self):
        self._cache.run()
        self._cache_thread.start()

    def close(self):
        self._cache.close()
