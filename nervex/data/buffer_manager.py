from abc import ABC, abstractmethod
import time
import os.path as osp
from threading import Thread
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np

from nervex.data.structure import ReplayBuffer, Cache, SumSegmentTree
from nervex.utils import deep_merge_dicts, remove_file
from nervex.config import buffer_manager_default_config

default_config = buffer_manager_default_config.replay_buffer


class IBuffer(ABC):

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def push(self, data: Union[list, dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, info: Dict[str, list]) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, cur_learner_iter: int) -> list:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, _state_dict: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def replay_start_size(self) -> int:
        raise NotImplementedError


class BufferManager(IBuffer):
    """
    Overview:
        Reinforcement Learning replay buffer's manager. Manage one or many buffers.
    Interface:
        __init__, push, sample, update, clear, count, start, close, state_dict, load_state_dict
    """

    def __init__(self, cfg: dict):
        """
        Overview:
            Initialize replay buffer
        Arguments:
            - cfg (:obj:``dict``): config dict
        """
        self.cfg = deep_merge_dicts(default_config, cfg)
        # ``buffer_name`` is a list containing all buffers' names
        self.buffer_name = self.cfg.buffer_name
        # ``buffer`` is a dict {buffer_name: prioritized_buffer}, where prioritized_buffer guarantees thread safety
        self.buffer = {}
        self._enable_track_used_data = {}
        self._delete_used_data_thread = {}
        for name in self.buffer_name:
            buffer_cfg = self.cfg[name]
            enable_track_used_data = buffer_cfg.get('enable_track_used_data', False)
            self.buffer[name] = ReplayBuffer(
                name=name,
                replay_buffer_size=buffer_cfg['replay_buffer_size'],
                replay_start_size=buffer_cfg.get('replay_start_size', 0),
                max_reuse=buffer_cfg.get('max_reuse', None),
                max_staleness=buffer_cfg.get('max_staleness', None),
                min_sample_ratio=buffer_cfg.get('min_sample_ratio', 1.),
                alpha=buffer_cfg.get('alpha', 0.),
                beta=buffer_cfg.get('beta', 0.),
                anneal_step=buffer_cfg.get('anneal_step', 0),
                enable_track_used_data=enable_track_used_data,
                deepcopy=buffer_cfg.get('deepcopy', False),
                monitor_cfg=buffer_cfg.get('monitor', None),
            )
            self._enable_track_used_data[name] = enable_track_used_data
            if self._enable_track_used_data[name]:
                self._delete_used_data_thread[name] = Thread(
                    target=self._delete_used_data, args=(name, ), name='delete_used_data'
                )

        # Cache mechanism: First push data into cache, then(on some conditions) put forward to meta buffer.
        # self.use_cache = cfg.get('use_cache', False)
        self.use_cache = False
        self._cache = Cache(maxlen=self.cfg.get('cache_maxlen', 256), timeout=self.cfg.get('timeout', 8))
        self._cache_thread = Thread(target=self._cache2meta, name='buffer_cache')
        self._end_flag = False

    def _cache2meta(self):
        """
        Overview:
            Get data from ``_cache`` and push it into ``_meta_buffer``
        """
        # loop until the end flag is sent to the cache(the close method of the cache)
        for data in self._cache.get_cached_data_iter():
            with self._meta_lock:
                self._meta_buffer.append(data)

    def push(self, data: Union[list, dict], buffer_name: Optional[List[str]] = None) -> None:
        """
        Overview:
            Push ``data`` into appointed buffer.
        Arguments:
            - data (:obj:``list`` or ``dict``): Data list or data item (dict type).
            - buffer_name (:obj:``Optional[List[str]]``): The buffer to push data into
        """
        assert (isinstance(data, list) or isinstance(data, dict)), type(data)
        if isinstance(data, dict):
            data = [data]
        if self.use_cache:
            for d in data:
                self._cache.push_data(d)
        else:
            if buffer_name is None:
                elem = data[0]
                buffer_name = elem.get('buffer_name', self.buffer_name)
            for n in buffer_name:
                self.buffer[n].extend(data)

    def sample(
            self,
            batch_size: Union[int, Dict[str, int]],
            cur_learner_iter: int,
    ) -> Union[list, Dict[str, list]]:
        """
        Overview:
            Sample data from prioritized buffers according to ``batch_size`.
        Arguments:
            - batch_size (:obj:``Union[int, Dict[str, int]]``): Batch size of the data that will be sampled. \
                Caller can indicate the corresponding batch_size when sampling from multiple buffers.
            - cur_learner_iter (:obj:``int``): Learner's current iteration, used to calculate staleness.
        Returns:
            - data (:obj:``Union[list, Dict[str, list]]``): Sampled data batch.
        """
        # single buffer case
        if isinstance(batch_size, int):
            assert len(self.buffer_name) == 1
            batch_size = {name: batch_size for name in self.buffer_name}
        # Different buffers' sample check and sample
        # buffer_sample_data is ``List[List[dict]]``, a list containing ``len(batch_size)`` lists which
        # contains datas sampled from corresponding buffer.
        buffer_sample_data = {}
        for buffer_name, sample_num in batch_size.items():
            assert buffer_name in self.buffer.keys(), '{}-{}'.format(buffer_name, self.buffer.keys())
            if not self.buffer[buffer_name].sample_check(sample_num, cur_learner_iter):
                return None
        for buffer_name, sample_num in batch_size.items():
            data = self.buffer[buffer_name].sample(sample_num, cur_learner_iter)
            buffer_sample_data[buffer_name] = data

        if len(buffer_sample_data) == 1:
            buffer_sample_data = list(buffer_sample_data.values())[0]
        return buffer_sample_data

    def update(self, info: Union[Dict[str, list], Dict[str, Dict[str, list]]]) -> None:
        """
        Overview:
            Update prioritized buffers with outside info. Current info includes transition's priority update.
        Arguments:
            - info (:obj:``Dict[str, list]``): Info dict. Currently contains keys \
                ['replay_unique_id', 'replay_buffer_idx', 'priority']. \
                "repaly_unique_id" format is "{buffer name}_{count in this buffer}"
        """
        # no priority case
        if info.get('priority', None) is None:
            return
        # single buffer case
        if not set(info.keys()).issubset(set(self.buffer_name)):
            assert len(self.buffer_name) == 1
            info = {name: info for name in self.buffer_name}
        for name, buffer_info in info.items():
            self.buffer[name].update(buffer_info)

    def clear(self, buffer_name: Optional[List[str]] = None) -> None:
        """
        Overview:
            Clear one replay buffer by excluding all data(including cache)
        Arguments:
            - buffer_name (:obj:``Optional[List[str]]``): Name of the buffer to be cleared.
        """
        # TODO(nyz) clear cache data
        if buffer_name is None:
            buffer_name = self.buffer_name
        for name in buffer_name:
            self.buffer[name].clear()

    def start(self) -> None:
        """
        Overview:
            Launch ``Cache`` thread and ``_cache2meta`` thread
        """
        for name, flag in self._enable_track_used_data.items():
            if flag:
                self._delete_used_data_thread[name].start()
        if self.use_cache:
            self._cache.run()
            self._cache_thread.start()
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Shut down the cache gracefully, as well as each buffer's tensorboard logger.
        """
        self._end_flag = True
        if self.use_cache:
            self._cache.close()
        for buffer in self.buffer.values():
            buffer.close()

    def count(self, buffer_name: Optional[str] = None) -> int:
        """
        Overview:
            Return chosen buffer's current data count.
        Arguments:
            - buffer_name (:obj:``Optional[str]``): Chosen buffer's name
        Returns:
            - count (:obj:``int``): Chosen buffer's data count
        """
        if buffer_name is None:
            validlen = [self.buffer[n].validlen for n in self.buffer_name]
            return min(validlen)
        else:
            return self.buffer[buffer_name].validlen

    def state_dict(self) -> dict:
        return {n: self.buffer[n].state_dict() for n in self.buffer_name}

    def load_state_dict(self, _state_dict: dict, strict: bool = True) -> None:
        if strict:
            assert set(_state_dict.keys()) == set(self.buffer.keys()
                                                  ), '{}/{}'.format(set(_state_dict.keys()), set(self.buffer.keys()))
        for n, v in _state_dict.items():
            if n in self.buffer.keys():
                self.buffer[n].load_state_dict(v)

    def replay_start_size(self, buffer_name: Optional[str] = None) -> int:
        if buffer_name is None:
            return max([self.buffer[n].replay_start_size for n in self.buffer_name])
        else:
            return self.buffer[buffer_name].replay_start_size

    def _delete_used_data(self, name: str) -> None:
        while not self._end_flag:
            data = self.buffer[name].used_data
            if data:
                remove_file(data)
            else:
                time.sleep(0.001)
