import copy
import os.path as osp
from threading import Thread
from typing import Union, Optional
from functools import partial
import time

from nervex.data.structure import PrioritizedBuffer, Cache
from nervex.utils import LockContext, LockContextType, read_config, deep_merge_dicts, EasyTimer
from nervex.torch_utils import build_log_buffer
from nervex.utils import build_logger, TextLogger, TensorBoardLogger, VariableRecord
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
    __thruput_property_names = ['in_count', 'out_count']

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.in_count = 0
        self.out_count = 0
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _sum = sum([_value for (_begin_time, _end_time), _value in records])
            return _sum / self.expire

        for _prop_name in self.__thruput_property_names:
            self.register_attribute_value('thruput', _prop_name, partial(__avg_func, prop_name=_prop_name))


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
    reuse_avg = LoggedValue(float)
    reuse_max = LoggedValue(int)
    priority_avg = LoggedValue(float)
    priority_max = LoggedValue(float)
    priority_min = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.out_time = 0.0
        self.reuse_avg = 0.0
        self.reuse_max = 0
        self.priority_avg = 0.0
        self.priority_max = 0.0
        self.priority_min = 0.0
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _sum = sum([_value for (_begin_time, _end_time), _value in records])
            return _sum / self.expire

        def __max_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return max(_list)

        def __min_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return min(_list)

        self.register_attribute_value('thruput', 'out_time', partial(__avg_func, prop_name='out_time'))
        self.register_attribute_value('reuse', 'reuse_avg', partial(__avg_func, prop_name='reuse_avg'))
        self.register_attribute_value('reuse', 'reuse_max', partial(__max_func, prop_name='reuse_avg'))
        self.register_attribute_value('priority', 'priority_avg', partial(__avg_func, prop_name='priority_avg'))
        self.register_attribute_value('priority', 'priority_max', partial(__max_func, prop_name='priority_max'))
        self.register_attribute_value('priority', 'priority_min', partial(__min_func, prop_name='priority_min'))


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
            _sum = sum([_value for (_begin_time, _end_time), _value in records])
            return _sum / self.expire

        self.register_attribute_value('thruput', 'in_time', partial(__avg_func, prop_name='in_time'))


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
        # main buffer
        self._meta_buffer = PrioritizedBuffer(
            maxlen=self.cfg.meta_maxlen,
            max_reuse=max_reuse,
            min_sample_ratio=self.cfg.min_sample_ratio,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            enable_track_used_data=self.cfg.enable_track_used_data
        )
        self._meta_lock = LockContext(type_=LockContextType.THREAD_LOCK)

        # cache mechanism: first push data into cache, then(some conditions) put forward to meta buffer
        self._cache = Cache(maxlen=self.cfg.cache_maxlen, timeout=self.cfg.timeout)
        self._cache_thread = Thread(target=self._cache2meta)

        # monitor & logger
        self._timer = EasyTimer()  # record in & out time
        self._natural_monitor = NaturalMonitor(NaturalTime(), expire=self.cfg.monitor.natural_expire)
        self._out_count = 0
        self._out_tick_time = TickTime()
        self._out_tick_monitor = OutTickMonitor(self._out_tick_time, expire=self.cfg.monitor.tick_expire)
        self._in_count = 0
        self._in_tick_time = TickTime()
        self._in_tick_monitor = InTickMonitor(self._in_tick_time, expire=10)
        self._logger = TextLogger(self.cfg.monitor.log_path, name='buffer_logger')
        self._tb_logger = TensorBoardLogger(self.cfg.monitor.log_path, name='buffer_logger')
        self._in_record = VariableRecord(self.cfg.monitor.log_freq)
        self._out_record = VariableRecord(self.cfg.monitor.log_freq)
        for logger in [self._tb_logger, self._in_record]:
            for var in ['in_count', 'in_time']:
                logger.register_var(var)
        for logger in [self._tb_logger, self._out_record]:
            for var in ['out_count', 'out_time', 'reuse_avg', 'reuse_max', 'priority_avg', 'priority_max',
                        'priority_min']:
                logger.register_var(var)
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

        def push(item: dict) -> None:
            # push one single data item into ``self._cache``
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

        with self._timer:
            if isinstance(data, list):
                self._natural_monitor.in_count = len(data)
                for d in data:
                    push(d)
            elif isinstance(data, dict):
                self._natural_monitor.in_count = 1
                push(data)
        self._in_tick_monitor.in_time = self._timer.value
        self._in_tick_time.step()
        in_dict = {
            'in_count': self._natural_monitor.thruput['in_count'](),
            'in_time': self._in_tick_monitor.thruput['in_time']()
        }
        self._in_record.update_var(in_dict)
        self._in_count += 1
        if self._in_count % self._log_freq == 0:
            self._logger.info("===Add In Buffer {} Times===".format(self._in_count))
            self._logger.info(self._in_record.get_vars_text())
            tb_keys = ['in_time', 'in_count']
            self._tb_logger.add_val_list(
                self._in_record.get_vars_tb_format(tb_keys, self._in_count, var_type='scalar'), viz_type='scalar'
            )

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
            with self._meta_lock:
                data = self._meta_buffer.sample(batch_size)
        if data is None:
            # no enough element for sampling
            return None
        data_count = len(data)
        self._natural_monitor.out_count = data_count
        self._out_tick_monitor.out_time = self._timer.value
        reuse_list = [a['reuse'] for a in data]
        reuse_avg, reuse_max = sum(reuse_list) / len(reuse_list), max(reuse_list)
        assert isinstance(reuse_max, int)
        priority_list = [a['priority'] for a in data]
        priority_avg, priority_max, priority_min = sum(priority_list) / len(priority_list), max(priority_list
                                                                                                ), min(priority_list)
        self._out_tick_monitor.reuse_avg = reuse_avg
        self._out_tick_monitor.reuse_max = reuse_max
        self._out_tick_monitor.priority_avg = priority_avg
        self._out_tick_monitor.priority_max = priority_max
        self._out_tick_monitor.priority_min = priority_min
        self._out_tick_time.step()
        out_dict = {
            'out_count': self._natural_monitor.thruput['out_count'](),
            'out_time': self._out_tick_monitor.thruput['out_time'](),
            'reuse_avg': self._out_tick_monitor.reuse['reuse_avg'](),
            'reuse_max': self._out_tick_monitor.reuse['reuse_max'](),
            'priority_avg': self._out_tick_monitor.priority['priority_avg'](),
            'priority_max': self._out_tick_monitor.priority['priority_max'](),
            'priority_min': self._out_tick_monitor.priority['priority_min']()
        }
        self._out_record.update_var(out_dict)
        self._out_count += 1
        if self._out_count % self._log_freq == 0:
            self._logger.info("===Read Buffer {} Times===".format(self._out_count))
            self._logger.info(self._out_record.get_vars_text())
            tb_keys = [
                'out_count', 'out_time', 'reuse_avg', 'reuse_max', 'priority_avg', 'priority_max', 'priority_min'
            ]
            self._tb_logger.add_val_list(
                self._out_record.get_vars_tb_format(tb_keys, self._in_count, var_type='scalar'), viz_type='scalar'
            )
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

    def run(self) -> None:
        """
        Overview:
            Launch ``Cache`` thread and ``_cache2meta`` thread
        """
        self._cache.run()
        self._cache_thread.start()

    def close(self) -> None:
        """
        Overview:
            Shut down the cache gracefully
        """
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
