from nervex.utils import remove_file
from queue import Queue
from typing import Union
from threading import Thread
import time
from functools import partial
import threading

from nervex.utils.autolog import LoggedValue, LoggedModel


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

    def __init__(self) -> None:
        self._used_data = Queue()
        self._delete_used_data_thread = Thread(target=self._delete_used_data, name='delete_used_data')
        self._delete_used_data_thread.daemon = True
        self._end_flag = True

    def start(self) -> None:
        self._end_flag = False
        self._delete_used_data_thread.start()

    def close(self) -> None:
        while not self._used_data.empty():
            data_id = self._used_data.get()
            remove_file(data_id)
        self._end_flag = True

    def add_used_data(self, data) -> None:
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
    use = LoggedValue(float)
    priority = LoggedValue(float)
    staleness = LoggedValue(float)

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

        self.register_attribute_value('avg', 'use', partial(__avg_func, prop_name='use'))
        self.register_attribute_value('max', 'use', partial(__max_func, prop_name='use'))
        self.register_attribute_value('avg', 'priority', partial(__avg_func, prop_name='priority'))
        self.register_attribute_value('max', 'priority', partial(__max_func, prop_name='priority'))
        self.register_attribute_value('min', 'priority', partial(__min_func, prop_name='priority'))
        self.register_attribute_value('avg', 'staleness', partial(__avg_func, prop_name='staleness'))
        self.register_attribute_value('max', 'staleness', partial(__max_func, prop_name='staleness'))


class PeriodicThruputMonitor:

    def __init__(self, cfg, logger, tb_logger) -> None:
        self._end_flag = False
        self._logger = logger
        self._tb_logger = tb_logger
        self._thruput_print_seconds = cfg.seconds
        self._thruput_print_times = 0
        self._thruput_start_time = time.time()
        self._push_data_count = 0
        self._sample_data_count = 0
        self._remove_data_count = 0
        self._valid_count = 0
        self._thruput_log_thread = threading.Thread(
            target=self._thrput_print_periodically, args=(), name='periodic_thruput_log'
        )
        self._thruput_log_thread.daemon = True
        self._thruput_log_thread.start()

    def _thrput_print_periodically(self) -> None:
        while not self._end_flag:
            time_passed = time.time() - self._thruput_start_time
            if time_passed >= self._thruput_print_seconds:
                self._logger.info('In the past {:.1f} seconds, buffer statistics is as follows:'.format(time_passed))
                count_dict = {
                    'pushed_in': self._push_data_count,
                    'sampled_out': self._sample_data_count,
                    'removed': self._remove_data_count,
                    'current_have': self._valid_count,
                }
                self._logger.print_vars_hor(count_dict)
                for k, v in count_dict.items():
                    self._tb_logger.add_scalar('buffer_{}_sec/'.format(self.name) + k, v, self._thruput_print_times)
                self._push_data_count = 0
                self._sample_data_count = 0
                self._remove_data_count = 0
                self._thruput_start_time = time.time()
                self._thruput_print_times += 1
            else:
                time.sleep(min(1, self._thruput_print_seconds * 0.2))

    def close(self) -> None:
        self._end_flag = True

    @property
    def push_data_count(self) -> int:
        return self._push_data_count

    @push_data_count.setter
    def push_data_count(self, count) -> None:
        self._push_data_count = count

    @property
    def sample_data_count(self) -> int:
        return self._sample_data_count

    @sample_data_count.setter
    def sample_data_count(self, count) -> None:
        self._sample_data_count = count

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
