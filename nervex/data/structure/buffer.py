import copy
import math
import time
import numbers
from queue import Queue
from typing import Union, NoReturn, Any, Optional, List
import pickle
import numpy as np
from functools import partial
from easydict import EasyDict

from nervex.data.structure.segment_tree import SumSegmentTree, MinSegmentTree
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from nervex.utils import LockContext, LockContextType, EasyTimer, build_logger


class NaturalMonitor(LoggedModel):
    """
    Overview:
        NaturalMonitor is to monitor how many pieces of data are added into and read out from buffer per second.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    in_count = LoggedValue(int)
    out_count = LoggedValue(int)

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
        OutTickMonitor is to monitor read-out indicators for ``expire`` times recent read-outs.
        Indicators include: read out time; average and max of read out data items' reuse; average, max and min of
        read out data items' priorityl; average and max of staleness.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    out_time = LoggedValue(float)
    reuse = LoggedValue(int)
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

        self.register_attribute_value('avg', 'out_time', partial(__avg_func, prop_name='out_time'))
        self.register_attribute_value('avg', 'reuse', partial(__avg_func, prop_name='reuse'))
        self.register_attribute_value('max', 'reuse', partial(__max_func, prop_name='reuse'))
        self.register_attribute_value('avg', 'priority', partial(__avg_func, prop_name='priority'))
        self.register_attribute_value('max', 'priority', partial(__max_func, prop_name='priority'))
        self.register_attribute_value('min', 'priority', partial(__min_func, prop_name='priority'))
        self.register_attribute_value('avg', 'staleness', partial(__avg_func, prop_name='staleness'))
        self.register_attribute_value('max', 'staleness', partial(__max_func, prop_name='staleness'))


class InTickMonitor(LoggedModel):
    """
    Overview:
        InTickMonitor is to monitor add-in indicators for ``expire`` times recent add-ins.
        Indicators include: add in time.
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


class RecordList(list):
    r"""
    Overview:
        A list which can track its old elements when they are set to a new one.
    Interface:
        __init__, __setitem__
    """

    def __init__(self, *args, **kwargs) -> None:
        r"""
        Overview:
            Additionally init a queue, which is used to track old replaced elements.
        """
        super(RecordList, self).__init__(*args, **kwargs)
        self._used_data = Queue()

    def __setitem__(self, idx: Union[numbers.Integral, slice], data: Any) -> None:
        r"""
        Overview:
            Additionally append the replaced element(if not None) at the back of ``self._used_data``
            to track it for further operation.
        Arguments:
            - idx (:obj:`Union[numbers.Integral, slice]`): one or many, indicating the index where \
                    the element will be replaced by a new one
            - data (:obj:`Any`): one/many(depending on ``idx``) pieces of data to be inserted to the list
        """
        if isinstance(idx, numbers.Integral):
            if self[idx] is not None:
                self._used_data.put(self[idx])
        elif isinstance(idx, slice):
            old_data = self[idx]
            for d in old_data:
                if d is not None:
                    self._used_data.put(d)
        else:
            raise TypeError("not support idx type: {}".format(type(idx)))
        super().__setitem__(idx, data)


class PrioritizedBuffer:
    r"""
    Overview:
        Prioritized buffer, can store and sample data.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe.
    Interface:
        __init__, append, extend, sample, update
    Property:
        maxlen, validlen, beta
    """

    def __init__(
            self,
            name: str,
            load_path: Optional[str] = None,
            maxlen: int = 10000,
            max_reuse: Optional[int] = None,
            max_staleness: Optional[int] = None,
            min_sample_ratio: float = 1.,
            alpha: float = 0.,
            beta: float = 0.,
            anneal_step: int = 0,
            enable_track_used_data: bool = False,
            deepcopy: bool = False,
            monitor_cfg: Optional[EasyDict] = None,
            eps: float = 0.01,
    ) -> int:
        r"""
        Overview:
            Initialize the buffer
        Arguments:
            - name (:obj:`str`): Buffer name, mainly used to generate unique data id and logger name.
            - load_path (:obj:`Optional[str]`): Demonstration data path. Buffer will use file data to initialize \
                ``self._data``. ``None`` means not to load data at the beginning.
            - maxlen (:obj:`int`): The maximum value of the buffer length. If ``load_path`` is not ``None``, it is \
                highly recommended to set ``maxlen`` no fewer than demonstration data's length.
            - max_reuse (:obj:`int` or None): The maximum reuse times of each element in buffer
            - min_sample_ratio (:obj:`float`): The minimum ratio restriction for sampling, only when \
                "current element number in buffer / sample size" is greater than this, can start sampling
            - alpha (:obj:`float`): How much prioritization is used (0: no prioritization, 1: full prioritization)
            - beta (:obj:`float`): How much correction is used (0: no correction, 1: full correction)
            - anneal_step (:obj:`int`): Anneal step for beta(beta -> 1)
            - enable_track_used_data (:obj:`bool`): Whether to track the used data or not
            - deepcopy (:obj:`bool`): Whether to deepcopy data when append/extend and sample data
            - monitor_cfg (:obj:`EasyDict`): Monitor's dict config
            - eps (:obj:`float`): A small positive number to avoid edge case
        """
        # TODO(nyz) remove elements according to priority
        # ``_data`` is the core data structure to store data (or data's reference/file path)
        # use RecordList if needs to track used data; otherwise use normal list
        self._enable_track_used_data = enable_track_used_data
        if self._enable_track_used_data:
            self._data = RecordList([None for _ in range(maxlen)])
        else:
            self._data = [None for _ in range(maxlen)]
        # current valid data count, indicating how many elements in ``self._data`` is valid.
        self._valid_count = 0
        # how many pieces of data have been pushed into this buffer, should be no less than ``_valid_count``
        self._push_count = 0
        # point to the position where next data can be inserted, i.e. last data's next position
        self.pointer = 0
        # used to generate a unique id for each data: if a new data is inserted, its unique id will be this
        self.next_unique_id = 0
        # {position_idx/pointer_idx: reuse_count}
        self._reuse_count = {idx: 0 for idx in range(maxlen)}
        # max priority till now, used to initizalize a data's priority if "priority" is not passed in with the data
        self.max_priority = 1.0
        # small positive number to avoid edge-case, e.g. avoid "priority" to be 0
        self._eps = eps
        # data check function list, used in ``append`` and ``extend``
        self.check_list = [lambda x: isinstance(x, dict)]
        # lock to guarantee thread safe
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

        self.name = name
        self.load_path = load_path
        self._maxlen = maxlen
        self.max_reuse = max_reuse if max_reuse is not None else np.inf
        self.max_staleness = max_staleness if max_staleness is not None else np.inf
        assert min_sample_ratio >= 1, min_sample_ratio
        self.min_sample_ratio = min_sample_ratio
        assert 0 <= alpha <= 1, alpha
        self.alpha = alpha
        assert 0 <= beta <= 1, beta
        self._beta = beta
        self._anneal_step = anneal_step
        if self._anneal_step != 0:
            self._beta_anneal_step = (1 - self._beta) / self._anneal_step
        self._deepcopy = deepcopy

        # ``sum segtree and min segtree are used for prioritized sample``
        # capacity needs to be the power of 2
        capacity = int(np.power(2, np.ceil(np.log2(self.maxlen))))
        # sum segtree and min segtree are used to sample data according to priority
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        # monitor & logger
        if monitor_cfg is None:
            monitor_cfg = EasyDict(
                {
                    'log_freq': 2000,
                    'log_path': './log/buffer/',
                    'natural_expire': 100,
                    'tick_expire': 100,
                }
            )
        self.monitor_cfg = monitor_cfg
        self._timer = EasyTimer()  # to record in & out time
        self._natural_monitor = NaturalMonitor(NaturalTime(), expire=self.monitor_cfg.natural_expire)
        self._out_count = 0  # sample out operation count
        self._out_tick_monitor = OutTickMonitor(TickTime(), expire=self.monitor_cfg.tick_expire)
        self._in_count = 0  # push in operation count
        self._in_tick_monitor = InTickMonitor(TickTime(), expire=self.monitor_cfg.tick_expire)
        self._logger, self._tb_logger = build_logger(self.monitor_cfg.log_path, self.name, True)
        self._in_vars = ['in_count_avg', 'in_time_avg']
        self._in_vars = [self.name + var for var in self._in_vars]
        self._out_vars = [
            'out_count_avg', 'out_time_avg', 'reuse_avg', 'reuse_max', 'priority_avg', 'priority_max', 'priority_min'
        ]
        self._out_vars = [self.name + var for var in self._out_vars]
        for var in self._in_vars + self._out_vars:
            self._tb_logger.register_var(var)
        self._log_freq = self.monitor_cfg.log_freq

        # load data from file if asked to do so
        if self.load_path is not None:
            with open(self.load_path, "rb+") as f:
                _demo_data = pickle.load(f)
            self.extend(_demo_data)

    def _set_weight(self, idx: int, data: Any) -> None:
        r"""
        Overview:
            Set the priority and tree weight of the input data
        Arguments:
            - idx (:obj:`int`): the index of the list where the data will be inserted
            - data (:obj:`Any`): the data which will be inserted
        """
        if 'priority' not in data.keys() or data['priority'] is None:
            data['priority'] = self.max_priority
        weight = data['priority'] ** self.alpha
        self.sum_tree[idx] = weight
        self.min_tree[idx] = weight

    def sample(self, size: int, cur_learner_iter: int) -> Optional[list]:
        r"""
        Overview:
            Sample data with length ``size``
        Arguments:
            - size (:obj:`int`): the number of the data that will be sampled
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness
        Returns:
            - sample_data (:obj:`list`): If check fails returns None; Otherwise returns a list with length ``size``, \
                and each data owns keys: original keys + ['IS', 'priority', 'replay_unique_id', 'replay_buffer_idx']
        """
        with self._lock:
            with self._timer:
                left_to_sample = size
                result = []
                while left_to_sample:
                    if not self._sample_check(left_to_sample):
                        return None
                    indices = self._get_indices(left_to_sample)
                    result += self._sample_with_indices(indices, cur_learner_iter, len(result) == 0)
                    left_to_sample = size - len(result)
                # Deepcopy ``result``'s same indice datas in case ``self._get_indices`` may get datas with
                # the same indices, i.e. the same datas would be sampled afterwards.
                for i in range(size):
                    tmp = []
                    for j in range(i + 1, size):
                        if id(result[i]) == id(result[j]):
                            tmp.append(j)
                    for j in tmp:
                        result[j] = copy.deepcopy(result[j])

            self._monitor_update_of_sample(result, self._timer.value)
            return result

    def append(self, ori_data: Any) -> None:
        r"""
        Overview:
            Append a data item into queue. It is forbidden for demonstration buffer to call this function.
            Add two keys in data:

                - replay_unique_id: the data item's unique id, using ``self.next_unique_id`` to generate it
                - replay_buffer_idx: the data item's position index in the queue, this position may had an \
                    old element but wass replaced by this input new one. using ``self.pointer`` to generate it
        Arguments:
            - ori_data (:obj:`Any`): the data which will be inserted
        """
        with self._lock:
            with self._timer:
                if self._deepcopy:
                    data = copy.deepcopy(ori_data)
                else:
                    data = ori_data
                try:
                    assert (self._data_check(data))
                except AssertionError:
                    # if data check fails, return without any operations
                    print('illegal data {}, reject it...'.format(type(data)))
                    return
                if self._data[self.pointer] is None:
                    self._valid_count += 1
                self._push_count += 1
                data['replay_unique_id'] = self._generate_id(self.next_unique_id)
                data['replay_buffer_idx'] = self.pointer
                self._set_weight(self.pointer, data)
                self._data[self.pointer] = data
                self._reuse_count[self.pointer] = 0
                self.pointer = (self.pointer + 1) % self._maxlen
                self.next_unique_id += 1

            self._monitor_update_of_push(1, self._timer.value)

    def extend(self, ori_data: List[Any]) -> None:
        r"""
        Overview:
            Extend a data list into queue. It is forbidden for demonstration buffer to call this function.
            Add two keys in each data item, you can reference ``append`` for details.
        Arguments:
            - ori_data (:obj:`T`): the data list
        """
        with self._lock:
            with self._timer:
                if self._deepcopy:
                    data = copy.deepcopy(ori_data)
                else:
                    data = ori_data
                check_result = [self._data_check(d) for d in data]
                # only keep data items that pass data_check
                valid_data = [d for d, flag in zip(data, check_result) if flag]
                length = len(valid_data)
                for i in range(length):
                    valid_data[i]['replay_unique_id'] = self._generate_id(self.next_unique_id + i)
                    valid_data[i]['replay_buffer_idx'] = (self.pointer + i) % self.maxlen
                    self._set_weight((self.pointer + i) % self.maxlen, valid_data[i])
                    if self._data[(self.pointer + i) % self.maxlen] is None:
                        self._valid_count += 1
                    self._push_count += 1
                # when updating ``_data`` and ``_reuse_count``, should consider two cases
                # regarding the relationship between "pointer + data length" and "queue max length" to check whether
                # data will exceed beyond queue's max length limitation
                if self.pointer + length <= self._maxlen:
                    self._data[self.pointer:self.pointer + length] = valid_data
                    for idx in range(self.pointer, self.pointer + length):
                        self._reuse_count[idx] = 0
                else:
                    data_start = self.pointer
                    valid_data_start = 0
                    residual_num = len(valid_data)
                    while True:
                        space = self._maxlen - data_start
                        L = min(space, residual_num)
                        self._data[data_start:data_start + L] = valid_data[valid_data_start:valid_data_start + L]
                        residual_num -= L
                        for i in range(data_start, data_start + L):
                            self._reuse_count[i] = 0
                        if residual_num <= 0:
                            break
                        else:
                            data_start = 0
                # update ``pointer`` and ``next_unique_id`` after the whole list is pushed into
                self.pointer = (self.pointer + length) % self._maxlen
                self.next_unique_id += length

            self._monitor_update_of_push(length, self._timer.value)

    def _monitor_update_of_push(self, add_count: int, add_time: float) -> None:
        self._natural_monitor.in_count = add_count
        self._in_tick_monitor.in_time = add_time
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

    def _monitor_update_of_sample(self, sample_data: int, sample_time: float) -> None:
        self._natural_monitor.out_count = len(sample_data)
        self._out_tick_monitor.out_time = sample_time
        reuse = sum([d['reuse'] for d in sample_data]) / len(sample_data)
        priority = sum([d['priority'] for d in sample_data]) / len(sample_data)
        staleness = sum([d['staleness'] for d in sample_data]) / len(sample_data)
        self._out_tick_monitor.reuse = int(reuse)
        self._out_tick_monitor.priority = priority
        self._out_tick_monitor.staleness = staleness
        self._out_tick_monitor.time.step()
        out_dict = {
            'out_count_avg': self._natural_monitor.avg['out_count'](),
            'out_time_avg': self._out_tick_monitor.avg['out_time'](),
            'reuse_avg': self._out_tick_monitor.avg['reuse'](),
            'reuse_max': self._out_tick_monitor.max['reuse'](),
            'priority_avg': self._out_tick_monitor.avg['priority'](),
            'priority_max': self._out_tick_monitor.max['priority'](),
            'priority_min': self._out_tick_monitor.min['priority'](),
            'staleness_avg': self._out_tick_monitor.avg['staleness'](),
            'staleness_max': self._out_tick_monitor.max['staleness'](),
        }
        self._out_count += 1
        if self._out_count % self._log_freq == 0:
            self._logger.info("===Read Buffer {} Times===".format(self._out_count))
            self._logger.print_vars(out_dict)
            self._tb_logger.print_vars(out_dict, self._out_count, 'scalar')

    def update(self, info: dict) -> None:
        r"""
        Overview:
            Update priority according to the id and idx
        Arguments:
            - info (:obj:`dict`): info dict containing all necessary keys for priority update
        """
        with self._lock:
            data = [info['replay_unique_id'], info['replay_buffer_idx'], info['priority']]
            for id_, idx, priority in zip(*data):
                # if the data still exists in the queue, then do the update operation
                if self._data[idx] is not None \
                        and self._data[idx]['replay_unique_id'] == id_:  # confirm the same transition(data)
                    assert priority >= 0, priority
                    self._data[idx]['priority'] = priority + self._eps
                    self._set_weight(idx, self._data[idx])
                    # update max priority
                    self.max_priority = max(self.max_priority, priority)

    def _data_check(self, d) -> bool:
        r"""
        Overview:
            Data legality check, using rules in ``self.check_list``
        Arguments:
            - d (:obj:`T`): the data which needs to be checked
        Returns:
            - result (:obj:`bool`): whether the data passes the check
        """
        # only the data passes all the check functions, would the check return True
        return all([fn(d) for fn in self.check_list])

    def _sample_check(self, size: int) -> bool:
        r"""
        Overview:
            Check whether the buffer satisfies the sample condition:
            current elements count / planning sample count >= min_sample_ratio, i.e. whether buffer is enough to sample.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - result (:obj:`bool`): Whether the buffer can sample
        """
        if self._valid_count / size < self.min_sample_ratio:
            print(
                "[INFO({})] no enough element for sample(expect: {}/current have: {}, min_sample_ratio: {})".format(
                    time.time(), size, self._valid_count, self.min_sample_ratio
                )
            )
            return False
        else:
            return True

    def _get_indices(self, size: int) -> list:
        r"""
        Overview:
            Get the sample index list according to the priority probability,
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - index_list (:obj:`list`): A list including all the sample indices
        """
        # average divide size intervals and sample from them
        intervals = np.array([i * 1.0 / size for i in range(size)])
        # uniform sample in each interval
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        # rescale to [0, S), where S is the sum of the total sum_tree
        mass *= self.sum_tree.reduce()
        # find prefix sum index to approximate sample with probability
        return [self.sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int) -> None:
        self._data[idx] = None
        self.sum_tree[idx] = self.sum_tree.neutral_element
        self.min_tree[idx] = self.min_tree.neutral_element
        self._valid_count -= 1

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int, first_sample: bool = True) -> list:
        r"""
        Overview:
            Sample data with ``indices``; If a data item is reused for too many times,
            remove it and update internal variables(sum_tree, min_tree, valid_count)
        Arguments:
            - indices (:obj:`List[int]`): A list including all the sample indices
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness
        Returns:
            - data (:obj:`list`) Sampled data
        """
        # calculate max weight for normalizing IS
        sum_tree_root = self.sum_tree.reduce()
        p_min = self.min_tree.reduce() / sum_tree_root
        max_weight = (self._valid_count * p_min) ** (-self._beta)
        data = []
        for idx in indices:
            # calculate staleness, if too stale, remove it and do not add it to the return data
            collect_iter = self._data[idx].get('collect_iter', cur_learner_iter + 1)
            if isinstance(collect_iter, list):
                # timestep transition's collect_iter is a list
                collect_iter = min(collect_iter)
            # ``staleness`` might be -1, means invalid,
            # e.g. actor does not report collecting model iter, it is demo buffer(data not generated by actor) etc.
            staleness = cur_learner_iter - collect_iter
            if staleness >= self.max_staleness:
                self._remove(idx)
                continue
            if self._deepcopy:
                # deepcopy data for avoiding interference
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            assert (copy_data is not None)
            copy_data['staleness'] = staleness
            # store reuse for outer monitor
            copy_data['reuse'] = self._reuse_count[idx]
            # get IS(importance sampling weight for gradient step)
            p_sample = self.sum_tree[copy_data['replay_buffer_idx']] / sum_tree_root
            weight = (self._valid_count * p_sample) ** (-self._beta)
            copy_data['IS'] = weight / max_weight
            data.append(copy_data)
            self._reuse_count[idx] += 1
        # remove the item whose "reuse count" is greater than max_reuse
        for idx in indices:
            if self._reuse_count[idx] > self.max_reuse:
                self._remove(idx)
        # anneal update beta, only the first sample will update, later samples caused by staleness will not update
        # because they are together belong to one actual sample
        if first_sample and self._anneal_step != 0:
            self._beta += self._beta_anneal_step
        return data

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variable
        """
        for i in range(len(self._data)):
            self._remove(i)
            self._reuse_count[i] = 0
        self._valid_count = 0
        self.pointer = 0
        self.max_priority = 1.0

    def _generate_id(self, data_id: int) -> str:
        """
        Overview:
            Use ``self.name`` and input ``id`` to generate a unique id for next data to be inserted.
        """
        return self.name + str(data_id)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def validlen(self) -> int:
        return self._valid_count

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> NoReturn:
        self._beta = beta

    @property
    def used_data(self) -> Any:
        if self._enable_track_used_data:
            if not self._data._used_data.empty():
                return self._data._used_data.get()
            else:
                return None
        else:
            return None

    @property
    def push_count(self) -> int:
        return self._push_count
