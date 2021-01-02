import copy
import os.path as osp
from threading import Thread
from typing import Union, Optional
from functools import partial
import time

from nervex.data.structure import PrioritizedBuffer, Cache
from nervex.utils import LockContext, LockContextType, read_config, deep_merge_dicts, EasyTimer
from nervex.utils import build_logger
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode

default_config = read_config(osp.join(osp.dirname(__file__), 'replay_buffer_default_config.yaml')).replay_buffer


class NaturalMonitor(LoggedModel):
    """
    Overview:
        NaturalMonitor is to monitor how many pieces of data are added to and read from buffer per second.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    in_count = LoggedValue(int)
    out_count = LoggedValue(int)

    # __thruput_property_names = ['in_count', 'out_count']

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _sum = sum([_value for (_begin_time, _end_time), _value in records])
            return _sum / self.expire

        self.register_attribute_value('avg', 'in_count', partial(__avg_func, prop_name='in_count'))
        self.register_attribute_value('avg', 'out_count', partial(__avg_func, prop_name='out_count'))


class OutTickMonitor(LoggedModel):
    """
    Overview:
        OutTickMonitor is to monitor read-out indices for ``expire`` times recent read-outs.
        Indices include: read out time; average and max of read out data items' reuse; average, max and min of
        read out data items' priority.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    out_time = LoggedValue(float)
    reuse = LoggedValue(int)
    priority = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        def __max_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return max(_list)

        def __min_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return min(_list)

        self.register_attribute_value('avg', 'out_time', partial(__avg_func, prop_name='out_time'))
        self.register_attribute_value('avg', 'reuse', partial(__avg_func, prop_name='reuse'))
        self.register_attribute_value('max', 'reuse', partial(__max_func, prop_name='reuse'))
        self.register_attribute_value('avg', 'priority', partial(__avg_func, prop_name='priority'))
        self.register_attribute_value('max', 'priority', partial(__max_func, prop_name='priority'))
        self.register_attribute_value('min', 'priority', partial(__min_func, prop_name='priority'))


class InTickMonitor(LoggedModel):
    """
    Overview:
        InTickMonitor is to monitor add-in indices for ``expire`` times recent add-ins.
        Indices include: add in time.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    in_time = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        self.register_attribute_value('avg', 'in_time', partial(__avg_func, prop_name='in_time'))


class ReplayBuffer:
    """
    Overview:
        Reinforcement Learning replay buffer, with priority sampling, data cache
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
        max_reuse = self.cfg.max_reuse if 'max_reuse' in self.cfg.keys() else None
        # traj_len is actor's generating trajectory length, often equals to or greater than actual data_push_length
        self.traj_len = cfg.get('traj_len', None)
        # unroll_len is learner's training data length, often smaller than traj_len
        self.unroll_len = cfg.get('unroll_len', None)
        self.use_cache = cfg.get('use_cache', False)
        # main buffer
        self._meta_buffer = PrioritizedBuffer(
            maxlen=self.cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=self.cfg.min_sample_ratio,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            anneal_step=self.cfg.anneal_step,
            enable_track_used_data=self.cfg.enable_track_used_data,
            deepcopy=self.cfg.deepcopy,
        )
        self._meta_lock = LockContext(type_=LockContextType.THREAD_LOCK)

        # cache mechanism: first push data into cache, then(some conditions) put forward to meta buffer
        self._cache = Cache(maxlen=self.cfg.cache_maxlen, timeout=self.cfg.timeout)
        self._cache_thread = Thread(target=self._cache2meta)

        # monitor & logger
        self._timer = EasyTimer()  # to record in & out time
        self._natural_monitor = NaturalMonitor(NaturalTime(), expire=self.cfg.monitor.natural_expire)
        self._out_count = 0
        self._out_tick_monitor = OutTickMonitor(TickTime(), expire=self.cfg.monitor.tick_expire)
        self._in_count = 0
        self._in_tick_monitor = InTickMonitor(TickTime(), expire=self.cfg.monitor.tick_expire)
        self._logger, self._tb_logger = build_logger(self.cfg.monitor.log_path, './log/buffer', True)
        self._in_vars = ['in_count_avg', 'in_time_avg']
        self._out_vars = [
            'out_count_avg', 'out_time_avg', 'reuse_avg', 'reuse_max', 'priority_avg', 'priority_max', 'priority_min'
        ]
        for var in self._in_vars + self._out_vars:
            self._tb_logger.register_var(var)
        self._log_freq = self.cfg.monitor.log_freq

    def _cache2meta(self):
        """
        Overview:
            Get data from ``_cache`` and push it into ``_meta_buffer``
        """
        # loop until the end flag is sent to the cache(the close method of the cache)
        for data in self._cache.get_cached_data_iter():
            with self._meta_lock:
                self._meta_buffer.append(data)

    def push_data(self, data: Union[list, dict]) -> None:
        """
        Overview:
            Push ``data`` into ``self._cache``
        Arguments:
            - data (:obj:`list` or `dict`): data list or data item (dict type)
        Note:
            thread-safe, because cache itself is thread-safe
        """
        assert (isinstance(data, list) or isinstance(data, dict))

        def split(item: dict) -> list:
            data_push_length = item['data_push_length']
            traj_len = self.traj_len if self.traj_len is not None else data_push_length
            unroll_len = self.unroll_len if self.unroll_len is not None else data_push_length
            assert data_push_length == traj_len
            split_num = traj_len // unroll_len
            split_item = [copy.deepcopy(item) for _ in range(split_num)]
            for i in range(split_num):
                split_item[i]['unroll_split_begin'] = i * unroll_len
                split_item[i]['unroll_len'] = unroll_len
            return split_item

        with self._timer:
            if isinstance(data, dict):
                data = [data]
            if 'data_push_length' in data[0].keys():
                split_data = []
                for d in data:
                    split_data += split(d)
            else:
                split_data = data
            if self.use_cache:
                for d in split_data:
                    self._cache.push_data(d)
            else:
                self._meta_buffer.extend(split_data)
        self._in_tick_monitor.in_time = self._timer.value
        self._in_tick_monitor.time.step()
        in_dict = {
            'in_count_avg': self._natural_monitor.avg['in_count'](),
            'in_time_avg': self._in_tick_monitor.avg['in_time']()
        }
        self._in_count += 1
        if self._in_count % self._log_freq == 0:
            self._logger.info("===Add In Buffer {} Times===".format(self._in_count))
            self._logger.print_vars(in_dict)
            self._tb_logger.print_vars(in_dict, self._in_count, 'scalar')

    def sample(self, batch_size: int) -> Optional[list]:
        """
        Overview:
            Sample data from replay buffer
        Arguments:
            - batch_size (:obj:`int`): the batch size of the data that will be sampled
        Returns:
            - data (:obj:`list` ): sampled data batch
        Note:
            thread-safe
        """
        with self._timer:
            sleep_count = 1
            while True:
                with self._meta_lock:
                    data = self._meta_buffer.sample(batch_size)
                if data is not None:
                    break
                time.sleep(sleep_count)
                sleep_count += 2
        data_count = len(data)
        self._natural_monitor.out_count = data_count
        self._out_tick_monitor.out_time = self._timer.value
        reuse = sum([d['reuse'] for d in data]) / batch_size
        priority = sum([d['priority'] for d in data]) / batch_size
        self._out_tick_monitor.reuse = int(reuse)
        self._out_tick_monitor.priority = priority
        self._out_tick_monitor.time.step()
        out_dict = {
            'out_count_avg': self._natural_monitor.avg['out_count'](),
            'out_time_avg': self._out_tick_monitor.avg['out_time'](),
            'reuse_avg': self._out_tick_monitor.avg['reuse'](),
            'reuse_max': self._out_tick_monitor.max['reuse'](),
            'priority_avg': self._out_tick_monitor.avg['priority'](),
            'priority_max': self._out_tick_monitor.max['priority'](),
            'priority_min': self._out_tick_monitor.min['priority'](),
        }
        self._out_count += 1
        if self._out_count % self._log_freq == 0:
            self._logger.info("===Read Buffer {} Times===".format(self._out_count))
            self._logger.print_vars(out_dict)
            self._tb_logger.print_vars(out_dict, self._out_count, 'scalar')
        return data

    def update(self, info: dict) -> None:
        """
        Overview:
            Update meta buffer with outside info
        Arguments:
            - info (:obj:`dict`): info dict
        Note:
            thread-safe
        """
        with self._meta_lock:
            self._meta_buffer.update(info)

    def clear(self) -> None:
        """
        Overview: clear replay buffer, exclude all the data(including cache)
        """
        # TODO(nyz) clear cache data
        self._meta_buffer.clear()

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

    @property
    def count(self) -> None:
        """
        Overview:
            Return buffer's current data count
        """
        return self._meta_buffer.validlen

    @property
    def used_data(self) -> 'Queue':  # noqa
        """
        Overview:
            Return the used data (used data means it was once in the buffer, but was replaced and discarded afterwards)
        """
        return self._meta_buffer.used_data
