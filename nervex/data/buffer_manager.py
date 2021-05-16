from abc import ABC, abstractmethod
import time
import copy
import os.path as osp
from threading import Thread
from typing import Union, Optional, Dict, Any, List, Tuple
from easydict import EasyDict
import numpy as np

from nervex.data.structure import PrioritizedReplayBuffer, NaiveReplayBuffer, Cache, SumSegmentTree
from nervex.utils import deep_merge_dicts, remove_file


class IBuffer(ABC):

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def push(self, data: Union[list, dict], cur_collector_envstep: int) -> None:
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
    def replay_buffer_start_size(self) -> int:
        raise NotImplementedError


class BufferManager(IBuffer):
    """
    Overview:
        Reinforcement Learning replay buffer's manager. Manage one or many buffers.
    Interface:
        __init__, push, sample, update, clear, count, start, close, state_dict, load_state_dict
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        # TODO(nyz) buffer type
        return PrioritizedReplayBuffer.default_config()

    def __init__(self, cfg: dict, tb_logger: Optional['SummaryWriter'] = None) -> None:  # noqa
        """
        Overview:
            Initialize replay buffer
        Arguments:
            - cfg (:obj:``dict``): config dict
        """
        # ``self.buffer_name`` is a list containing all buffers' names
        if 'buffer_name' in cfg:
            self.buffer_name = cfg['buffer_name']
        else:
            self.buffer_name = ['agent']
            cfg.buffer_type = 'priority'
        self.cfg = {}
        for name in self.buffer_name:
            if name in cfg:
                self.cfg[name] = cfg.pop(name)
            else:
                self.cfg[name] = cfg
        # ``self.buffer`` is a dict {buffer_name: prioritized_buffer}, where prioritized_buffer guarantees thread safety
        self.buffer = {}
        self._enable_track_used_data = {}
        self._delete_used_data_thread = {}
        for name in self.buffer_name:
            buffer_cfg = self.cfg[name]
            buffer_type = buffer_cfg.buffer_type
            if buffer_type == 'priority':
                buffer_cls = PrioritizedReplayBuffer
            elif buffer_type == 'naive':
                buffer_cls = NaiveReplayBuffer
            else:
                raise TypeError("invalid buffer type: {}".format(buffer_type))
            self.buffer[name] = buffer_cls(
                name=name,
                cfg=buffer_cfg,
                tb_logger=tb_logger,
            )
            enable_track_used_data = buffer_cfg.enable_track_used_data
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

    def push(
            self,
            data: Union[list, dict],
            buffer_name: Optional[List[str]] = None,
            cur_collector_envstep: int = -1
    ) -> None:
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
            for i, n in enumerate(buffer_name):
                # TODO optimizer multi-buffer deepcopy
                if i >= 1:
                    self.buffer[n].extend(copy.deepcopy(data), cur_collector_envstep)
                else:
                    self.buffer[n].extend(data, cur_collector_envstep)

    def sample(
            self,
            batch_size: Union[int, Dict[str, int]],
            cur_learner_iter: int,
    ) -> Union[list, Dict[str, list]]:
        """
        Overview:
            Sample data from prioritized buffers according to ``batch_size``.
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

    def __del__(self):
        if not self._end_flag:
            self.close()

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

    def replay_buffer_start_size(self, buffer_name: Optional[str] = None) -> int:
        if buffer_name is None:
            return max([self.buffer[n].replay_buffer_start_size for n in self.buffer_name])
        else:
            return self.buffer[buffer_name].replay_buffer_start_size

    def _delete_used_data(self, name: str) -> None:
        while not self._end_flag:
            data = self.buffer[name].used_data
            if data:
                remove_file(data)
            else:
                time.sleep(0.001)
