import os.path as osp
from threading import Thread
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np

from nervex.data.structure import PrioritizedBuffer, Cache, SumSegmentTree
from nervex.utils import read_config, deep_merge_dicts

default_config = read_config(osp.join(osp.dirname(__file__), 'replay_buffer_default_config.yaml')).replay_buffer


class ReplayBuffer:
    """
    Overview:
        Reinforcement Learning replay buffer, with prioritized sampling and data cache.
    Interface:
        __init__, push_data, sample, update, run, close
    """

    def __init__(self, cfg: dict):
        """
        Overview:
            Initialize replay buffer
        Arguments:
            - cfg (:obj:`dict`): config dict
        """
        self.cfg = deep_merge_dicts(default_config, cfg)
        # ``buffer_name``` is a list containing all buffers' names
        self.buffer_name = self.cfg.buffer_name
        # ``buffer`` is a dict {buffer_name: prioritized_buffer}, where prioritized_buffer guarantees thread safety
        self.buffer = {}
        for name in self.buffer_name:
            buffer_cfg = self.cfg[name]
            self.buffer[name] = PrioritizedBuffer(
                name=name,
                load_path=buffer_cfg.get('load_path', None),
                maxlen=buffer_cfg.get('maxlen', 10000),
                max_reuse=buffer_cfg.get('max_reuse', None),
                max_staleness=buffer_cfg.get('max_staleness', None),
                min_sample_ratio=buffer_cfg.get('min_sample_ratio', 1.),
                alpha=buffer_cfg.get('alpha', 0.),
                beta=buffer_cfg.get('beta', 0.),
                anneal_step=buffer_cfg.get('anneal_step', 0),
                enable_track_used_data=buffer_cfg.get('enable_track_used_data', False),
                deepcopy=buffer_cfg.get('deepcopy', False),
                monitor_cfg=buffer_cfg.get('monitor', None),
            )

        self.sample_tree = SumSegmentTree(len(self.buffer_name))
        for idx, name in enumerate(self.buffer_name):
            self.sample_tree[idx] = self.cfg.sample_ratio[name]
        assert self.sample_tree.reduce() == 1

        # cache mechanism: first push data into cache, then(some conditions) put forward to meta buffer
        # self.use_cache = cfg.get('use_cache', False)
        self.use_cache = False
        self._cache = Cache(maxlen=self.cfg.get('cache_maxlen', 256), timeout=self.cfg.get('timeout', 8))
        self._cache_thread = Thread(target=self._cache2meta)

    def _transform_sample_ratio(self, sample_ratio: dict) -> List[Tuple[str, float]]:
        # transform cfg.sample_ratio to a list, and calculate the prefix sum for sampling
        sample_ratio = [(name, ratio) for name, ratio in sample_ratio.items()]
        for idx in range(1, len(sample_ratio)):
            sample_ratio[idx][1] += sample_ratio[idx - 1][1]
        return sample_ratio

    def _cache2meta(self):
        """
        Overview:
            Get data from ``_cache`` and push it into ``_meta_buffer``
        """
        # loop until the end flag is sent to the cache(the close method of the cache)
        for data in self._cache.get_cached_data_iter():
            with self._meta_lock:
                self._meta_buffer.append(data)

    def push_data(self, data: Union[list, dict], buffer_name: str = "agent") -> None:
        """
        Overview:
            Push ``data`` into appointed buffer.
        Arguments:
            - data (:obj:`list` or `dict`): Data list or data item (dict type).
            - buffer_name (:obj:`str`): The buffer to push data into, default set to "agent".
        """
        assert (isinstance(data, list) or isinstance(data, dict))
        if isinstance(data, dict):
            data = [data]
        if self.use_cache:
            for d in data:
                self._cache.push_data(d)
        else:
            self.buffer[buffer_name].extend(data)

    def sample(self,
               batch_size: int,
               cur_learner_iter: int,
               sample_ratio: Optional[Dict[str, int]] = None) -> Optional[list]:
        """
        Overview:
            Sample data from prioritized buffers according to sample ratio.
        Arguments:
            - batch_size (:obj:`int`): Batch size of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
            - sample_ratio (:obj:`Optional[Dict[str, int]]`): How to sample from multiple buffers. Caller can pass \
                this argument; If not, replay buffer will use its own attribute ``self.sample_ratio``.
        Returns:
            - data (:obj:`list` ): Sampled data batch
        Note:
            thread-safe
        """
        if sample_ratio is not None:
            for idx, name in enumerate(self.buffer_name):
                self.sample_tree[idx] = sample_ratio[name]
            assert self.sample_tree.reduce() == 1

        # average divide size intervals and sample from them
        intervals = np.array([i * 1.0 / batch_size for i in range(batch_size)])
        # uniform sample in each interval
        mass = intervals + np.random.uniform(size=(batch_size, )) * 1. / batch_size
        # find prefix sum index to approximate sample with probability
        buffer_choice = [self.sample_tree.find_prefixsum_idx(m) for m in mass]

        # buffer_data is ``List[List[dict]]``, a list containing ``buffer_num`` lists which contains datas sampled from
        # this corresponding buffer.
        buffer_data = []
        buffer_num = len(self.buffer_name)
        for buffer_idx in range(buffer_num):
            size = buffer_choice.count(buffer_idx)
            data = self.buffer[self.buffer_name[buffer_idx]].sample(size, cur_learner_iter)
            if data is None:
                buffer_choice = [i if i != buffer_idx else -1 for i in buffer_choice]
            buffer_data.append(data)
        if not any(buffer_data):
            # all elements in buffer is None
            print('all elements in buffer is None')
            return None
        # todo: what if any(not all) buffer data is None

        data = [None for _ in range(batch_size)]
        # fill ``data`` with datas from agent buffer and demo buffer according to ``data_source``
        for data_idx, buffer_idx in enumerate(buffer_choice):
            if buffer_idx == -1:
                continue
            data[data_idx] = buffer_data[buffer_idx].pop()
        data = [d for d in data if d is not None]
        assert len(data) != 0

        return data

    def update(self, info: Dict[str, Any]) -> None:
        """
        Overview:
            Update prioritized buffers with outside info. Current info includes transition's priority update.
        Arguments:
            - info (:obj:`Dict[str, Any]`): Info dict. Currently contains keys \
                ['replay_unique_id', 'replay_buffer_idx', 'priority']
        """
        for name, buffer in self.buffer.items():
            buffer.update(info)

    def clear(self, buffer_name: Optional[List[str]] = None) -> None:
        """
        Overview:
            Clear prioritized buffer, exclude all data(including cache)
        """
        # TODO(nyz) clear cache data
        if buffer_name is None:
            buffer_name = self.buffer_name
        for name in buffer_name:
            self.buffer[name].clear()

    def run(self) -> None:
        """
        Overview:
            Launch ``Cache`` thread and ``_cache2meta`` thread
        """
        if self.use_cache:
            self._cache.run()
            self._cache_thread.start()

    def close(self) -> None:
        """
        Overview:
            Shut down the cache gracefully
        """
        if self.use_cache:
            self._cache.close()

    def count(self, buffer_name: str = "agent") -> int:
        """
        Overview:
            Return chosen buffer's current data count.
        Arguments:
            - buffer_name (:obj:`str`): Chosen buffer's name, default set to "agent"
        Returns:
            - count (:obj:`int`): Chosen buffer's data count
        """
        return self.buffer[buffer_name].validlen

    def used_data(self, buffer_name: str = "agent") -> 'queue.Queue':  # noqa
        """
        Overview:
            Return chosen buffer's "used data", which was once in the buffer, but was replaced and discarded afterwards
        Arguments:
            - buffer_name (:obj:`str`): Chosen buffer's name, default set to "agent"
        Returns:
            - queue (:obj:`queue.Queue`): Chosen buffer's record list
        """
        return self.buffer[buffer_name].used_data
