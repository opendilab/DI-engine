from typing import Any
import time
from queue import Queue
from typing import Union, Tuple
from threading import Thread
from functools import partial

from ding.utils.autolog import LoggedValue, LoggedModel
from ding.utils import LockContext, LockContextType, remove_file


def generate_id(name, data_id: int) -> str:
    """
    Overview:
        Use ``self.name`` and input ``id`` to generate a unique id for next data to be inserted.
    Arguments:
        - data_id (:obj:`int`): Current unique id.
    Returns:
        - id (:obj:`str`): Id in format "BufferName_DataId".
    """
    return "{}_{}".format(name, str(data_id))


class UsedDataRemover:
    """
    Overview:
        UsedDataRemover is a tool to remove file datas that will no longer be used anymore.
    Interface:
        start, close, add_used_data
    """

    def __init__(self) -> None:
        self._used_data = Queue()
        self._delete_used_data_thread = Thread(target=self._delete_used_data, name='delete_used_data')
        self._delete_used_data_thread.daemon = True
        self._end_flag = True

    def start(self) -> None:
        """
        Overview:
            Start the `delete_used_data` thread.
        """
        self._end_flag = False
        self._delete_used_data_thread.start()

    def close(self) -> None:
        """
        Overview:
            Delete all datas in `self._used_data`. Then join the `delete_used_data` thread.
        """
        while not self._used_data.empty():
            data_id = self._used_data.get()
            remove_file(data_id)
        self._end_flag = True

    def add_used_data(self, data: Any) -> None:
        """
        Overview:
            Delete all datas in `self._used_data`. Then join the `delete_used_data` thread.
        Arguments:
            - data (:obj:`Any`): Add a used data item into `self._used_data` for further remove.
        """
        assert data is not None and isinstance(data, dict) and 'data_id' in data
        self._used_data.put(data['data_id'])

    def _delete_used_data(self) -> None:
        while not self._end_flag:
            if not self._used_data.empty():
                data_id = self._used_data.get()
                remove_file(data_id)
            else:
                time.sleep(0.001)


class SampledDataAttrMonitor(LoggedModel):
    """
    Overview:
        SampledDataAttrMonitor is to monitor read-out indicators for ``expire`` times recent read-outs.
        Indicators include: read out time; average and max of read out data items' use; average, max and min of
        read out data items' priorityl; average and max of staleness.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    use_max = LoggedValue(int)
    use_avg = LoggedValue(float)
    priority_max = LoggedValue(float)
    priority_avg = LoggedValue(float)
    priority_min = LoggedValue(float)
    staleness_max = LoggedValue(int)
    staleness_avg = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list) if len(_list) != 0 else 0

        def __max_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return max(_list) if len(_list) != 0 else 0

        def __min_func(prop_name: str) -> Union[float, int]:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return min(_list) if len(_list) != 0 else 0

        self.register_attribute_value('avg', 'use', partial(__avg_func, prop_name='use_avg'))
        self.register_attribute_value('max', 'use', partial(__max_func, prop_name='use_max'))
        self.register_attribute_value('avg', 'priority', partial(__avg_func, prop_name='priority_avg'))
        self.register_attribute_value('max', 'priority', partial(__max_func, prop_name='priority_max'))
        self.register_attribute_value('min', 'priority', partial(__min_func, prop_name='priority_min'))
        self.register_attribute_value('avg', 'staleness', partial(__avg_func, prop_name='staleness_avg'))
        self.register_attribute_value('max', 'staleness', partial(__max_func, prop_name='staleness_max'))


class PeriodicThruputMonitor:
    """
    Overview:
        PeriodicThruputMonitor is a tool to record and print logs(text & tensorboard) how many datas are
        pushed/sampled/removed/valid in a period of time. For tensorboard, you can view it in 'buffer_{$NAME}_sec'.
    Interface:
        close
    Property:
        push_data_count, sample_data_count, remove_data_count, valid_count

    .. note::
        `thruput_log` thread is initialized and started in `__init__` method, so PeriodicThruputMonitor only provide
        one signle interface `close`
    """

    def __init__(self, name, cfg, logger, tb_logger) -> None:
        self.name = name
        self._end_flag = False
        self._logger = logger
        self._tb_logger = tb_logger
        self._thruput_print_seconds = cfg.seconds
        self._thruput_print_times = 0
        self._thruput_start_time = time.time()
        self._history_push_count = 0
        self._history_sample_count = 0
        self._remove_data_count = 0
        self._valid_count = 0
        self._thruput_log_thread = Thread(target=self._thrput_print_periodically, args=(), name='periodic_thruput_log')
        self._thruput_log_thread.daemon = True
        self._thruput_log_thread.start()

    def _thrput_print_periodically(self) -> None:
        while not self._end_flag:
            time_passed = time.time() - self._thruput_start_time
            if time_passed >= self._thruput_print_seconds:
                self._logger.info('In the past {:.1f} seconds, buffer statistics is as follows:'.format(time_passed))
                count_dict = {
                    'pushed_in': self._history_push_count,
                    'sampled_out': self._history_sample_count,
                    'removed': self._remove_data_count,
                    'current_have': self._valid_count,
                }
                self._logger.info(self._logger.get_tabulate_vars_hor(count_dict))
                for k, v in count_dict.items():
                    self._tb_logger.add_scalar('{}_sec/'.format(self.name) + k, v, self._thruput_print_times)
                self._history_push_count = 0
                self._history_sample_count = 0
                self._remove_data_count = 0
                self._thruput_start_time = time.time()
                self._thruput_print_times += 1
            else:
                time.sleep(min(1, self._thruput_print_seconds * 0.2))

    def close(self) -> None:
        """
        Overview:
            Join the `thruput_log` thread by setting `self._end_flag` to `True`.
        """
        self._end_flag = True

    def __del__(self) -> None:
        self.close()

    @property
    def push_data_count(self) -> int:
        return self._history_push_count

    @push_data_count.setter
    def push_data_count(self, count) -> None:
        self._history_push_count = count

    @property
    def sample_data_count(self) -> int:
        return self._history_sample_count

    @sample_data_count.setter
    def sample_data_count(self, count) -> None:
        self._history_sample_count = count

    @property
    def remove_data_count(self) -> int:
        return self._remove_data_count

    @remove_data_count.setter
    def remove_data_count(self, count) -> None:
        self._remove_data_count = count

    @property
    def valid_count(self) -> int:
        return self._valid_count

    @valid_count.setter
    def valid_count(self, count) -> None:
        self._valid_count = count


class ThruputController:

    def __init__(self, cfg) -> None:
        self._push_sample_rate_limit = cfg.push_sample_rate_limit
        assert 'min' in self._push_sample_rate_limit and self._push_sample_rate_limit['min'] >= 0
        assert 'max' in self._push_sample_rate_limit and self._push_sample_rate_limit['max'] <= float("inf")
        window_seconds = cfg.window_seconds
        self._decay_factor = 0.01 ** (1 / window_seconds)

        self._push_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._sample_lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._history_push_count = 0
        self._history_sample_count = 0

        self._end_flag = False
        self._count_decay_thread = Thread(target=self._count_decay, name='count_decay')
        self._count_decay_thread.daemon = True
        self._count_decay_thread.start()

    def _count_decay(self) -> None:
        while not self._end_flag:
            time.sleep(1)
            with self._push_lock:
                self._history_push_count *= self._decay_factor
            with self._sample_lock:
                self._history_sample_count *= self._decay_factor

    def can_push(self, push_size: int) -> Tuple[bool, str]:
        if abs(self._history_sample_count) < 1e-5:
            return True, "Can push because `self._history_sample_count` < 1e-5"
        rate = (self._history_push_count + push_size) / self._history_sample_count
        if rate > self._push_sample_rate_limit['max']:
            return False, "push({}+{}) / sample({}) > limit_max({})".format(
                self._history_push_count, push_size, self._history_sample_count, self._push_sample_rate_limit['max']
            )
        return True, "Can push."

    def can_sample(self, sample_size: int) -> Tuple[bool, str]:
        rate = self._history_push_count / (self._history_sample_count + sample_size)
        if rate < self._push_sample_rate_limit['min']:
            return False, "push({}) / sample({}+{}) < limit_min({})".format(
                self._history_push_count, self._history_sample_count, sample_size, self._push_sample_rate_limit['min']
            )
        return True, "Can sample."

    def close(self) -> None:
        self._end_flag = True

    @property
    def history_push_count(self) -> int:
        return self._history_push_count

    @history_push_count.setter
    def history_push_count(self, count) -> None:
        with self._push_lock:
            self._history_push_count = count

    @property
    def history_sample_count(self) -> int:
        return self._history_sample_count

    @history_sample_count.setter
    def history_sample_count(self, count) -> None:
        with self._sample_lock:
            self._history_sample_count = count
