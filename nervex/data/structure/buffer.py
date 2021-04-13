import copy
import logging
import math
import time
import numbers
from queue import Queue
from typing import Union, NoReturn, Any, Optional, List, Dict
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
        Indicators include: read out time; average and max of read out data items' use; average, max and min of
        read out data items' priorityl; average and max of staleness.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    out_time = LoggedValue(float)
    # use, priority and staleness are all averaged across one batch.
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

        self.register_attribute_value('avg', 'out_time', partial(__avg_func, prop_name='out_time'))
        self.register_attribute_value('avg', 'use', partial(__avg_func, prop_name='use'))
        self.register_attribute_value('max', 'use', partial(__max_func, prop_name='use'))
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


class ReplayBuffer:
    r"""
    Overview:
        Prioritized replay buffer, can store and sample data.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe, which means that functions like
        ``sample_check``, ``sample``, ``append``, ``extend``, ``clear`` are all mutual to each other.
    Interface:
        __init__, append, extend, sample, update
    Property:
        replay_buffer_size, validlen, beta
    """

    def __init__(
            self,
            name: str,
            replay_buffer_size: int = 10000,
            replay_start_size: int = 0,
            max_use: Optional[int] = None,
            max_staleness: Optional[int] = None,
            min_sample_ratio: float = 1.,
            alpha: float = 0.6,
            beta: float = 0.4,
            anneal_step: int = int(1e5),
            enable_track_used_data: bool = False,
            deepcopy: bool = False,
            monitor_cfg: Optional[EasyDict] = None,
            eps: float = 0.01,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
    ) -> int:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - name (:obj:`str`): Buffer name, mainly used to generate unique data id and logger name.
            - replay_buffer_size (:obj:`int`): The maximum value of the buffer length.
            - replay_start_size (:obj:`int`): The number of data in buffer when start training
            - max_use (:obj:`int` or None): The maximum use times of each element in buffer. Once a data is \
                sampled(used) ``max_use`` times, it would be removed out of buffer.
            - min_sample_ratio (:obj:`float`): The minimum ratio restriction for sampling, only when \
                "current element number in buffer" / "sample size" is greater than this, can start sampling.
            - alpha (:obj:`float`): How much prioritization is used (0: no prioritization, 1: full prioritization).
            - beta (:obj:`float`): How much correction is used (0: no correction, 1: full correction).
            - anneal_step (:obj:`Optional[Union[int, float]]`): Anneal step for beta, i.e. Beta takes \
                how many steps to come to 1. ``float("inf")`` means no annealing. Here, one step means \
                training for one iteration, i.e. sampling from buffer once.
            - enable_track_used_data (:obj:`bool`): Whether to track the used data or not.
            - deepcopy (:obj:`bool`): Whether to deepcopy data when append/extend and sample data.
            - monitor_cfg (:obj:`EasyDict`): Monitor's dict config.
            - eps (:obj:`float`): A small positive number to avoid edge case.
        """
        # TODO(nyz) remove elements according to priority
        # ``_data`` is a circular queue to store data (or data's reference/file path)
        self._data = [None for _ in range(replay_buffer_size)]
        self._enable_track_used_data = enable_track_used_data
        if self._enable_track_used_data:
            self._used_data = Queue()
            self._using_data = set()
            self._using_used_data = set()
        # Current valid data count, indicating how many elements in ``self._data`` is valid.
        self._valid_count = 0
        # How many pieces of data have been pushed into this buffer, should be no less than ``_valid_count``.
        self._push_count = 0
        # Point to the tail position where next data can be inserted, i.e. latest inserted data's next position.
        self._tail = 0
        # Point to the head of the circular queue. The true data is the stalest(oldest) data in this queue.
        # Because buffer would remove data due to staleness or use times, and at the beginning when queue is not
        # filled with data head would always be 0, so ``head`` may be not equal to ``tail``;
        # Otherwise, they two should be the same. Head is used to optimize staleness check in ``sample_check``.
        self._head = 0
        # Is used to generate a unique id for each data: If a new data is inserted, its unique id will be this.
        self._next_unique_id = 0
        # {position_idx: use_count}
        self._use_count = {idx: 0 for idx in range(replay_buffer_size)}
        # Max priority till now. Is used to initizalize a data's priority if "priority" is not passed in with the data.
        self._max_priority = 1.0
        # A small positive number to avoid edge-case, e.g. "priority" == 0.
        self._eps = eps
        # Data check function list, used in ``append`` and ``extend``. This buffer requires data to be dict.
        self.check_list = [lambda x: isinstance(x, dict)]
        # Lock to guarantee thread safe
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

        self.name = name
        self._replay_buffer_size = replay_buffer_size
        self._replay_start_size = replay_start_size
        self._max_use = max_use if max_use is not None else float("inf")
        self._max_staleness = max_staleness if max_staleness is not None else float("inf")
        assert min_sample_ratio >= 1, min_sample_ratio
        self.min_sample_ratio = min_sample_ratio
        assert 0 <= alpha <= 1, alpha
        self.alpha = alpha
        assert 0 <= beta <= 1, beta
        self._beta = beta
        self._anneal_step = anneal_step
        self._beta_anneal_one_step = (1 - self._beta) / self._anneal_step
        self._deepcopy = deepcopy

        # Prioritized sample.
        # Capacity needs to be the power of 2.
        capacity = int(np.power(2, np.ceil(np.log2(self.replay_buffer_size))))
        # Sum segtree and min segtree are used to sample data according to priority.
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)

        # Monitor & Logger
        if monitor_cfg is None:
            monitor_cfg = EasyDict(
                {
                    'log_freq': 2000,
                    'log_path': './log/buffer/default',
                    'natural_expire': 100,
                    'tick_expire': 100,
                }
            )
        self.monitor_cfg = monitor_cfg
        # To record in & out time.
        self._timer = EasyTimer()
        self._natural_monitor = NaturalMonitor(NaturalTime(), expire=self.monitor_cfg.natural_expire)
        # Sample out operation count.
        self._out_count = 0
        self._out_tick_monitor = OutTickMonitor(TickTime(), expire=self.monitor_cfg.tick_expire)
        # Add in operation count.
        self._in_count = 0
        self._in_tick_monitor = InTickMonitor(TickTime(), expire=self.monitor_cfg.tick_expire)
        if tb_logger is not None:
            self._logger, _ = build_logger(self.monitor_cfg.log_path, self.name + '_buffer', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                self.monitor_cfg.log_path,
                self.name + '_buffer',
            )
        self._log_freq = self.monitor_cfg.log_freq
        self._cur_learner_iter = -1
        self._cur_collector_envstep = -1

    def sample_check(self, size: int, cur_learner_iter: int) -> bool:
        r"""
        Overview:
            Do preparations for sampling and check whther data is enough for sampling
            Preparation includes removing stale transition in ``self._data``.
            Check includes judging whether this buffer satisfies the sample condition:
            current elements count / planning sample count >= min_sample_ratio.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.

        .. note::
            This function must be called exactly before calling ``sample``.
        """
        if size == 0:
            return True
        with self._lock:
            p = self._head
            while True:
                if self._data[p] is not None:
                    staleness = self._calculate_staleness(p, cur_learner_iter)
                    if staleness >= self._max_staleness:
                        self._remove(p)
                    else:
                        # Since the circular queue ``self._data`` guarantees that data's staleness is decreasing from
                        # index self._tail to index self._tail - 1, we can jump out of the loop as soon as
                        # meeting a fresh enough data
                        self._head = p
                        break
                p = (p + 1) % self._replay_buffer_size
                if p == self._tail:
                    # Traverse a circle and go back to the tail, which means can stop staleness checking now
                    break
            if self._valid_count / size < self.min_sample_ratio:
                self._logger.info(
                    "No enough elements for sampling (expect: {}/current have: {}, min_sample_ratio: {})".format(
                        size, self._valid_count, self.min_sample_ratio
                    )
                )
                return False
            else:
                return True

    def sample(self, size: int, cur_learner_iter: int) -> Optional[list]:
        r"""
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - sample_data (:obj:`list`): If check fails returns None; Otherwise returns a list with length ``size``, \
                and each data owns keys: original keys + ['IS', 'priority', 'replay_unique_id', 'replay_buffer_idx'].

        .. note::
            Before calling this function, ``sample_check`` must be called.
        """
        if size == 0:
            return []
        with self._lock:
            with self._timer:
                indices = self._get_indices(size)
                result = self._sample_with_indices(indices, cur_learner_iter)
                # Deepcopy ``result``'s same indice datas in case ``self._get_indices`` may get datas with
                # the same indices, i.e. the same datas would be sampled afterwards.
                for i in range(size):
                    tmp = []
                    for j in range(i + 1, size):
                        if id(result[i]) == id(result[j]):
                            tmp.append(j)
                    for j in tmp:
                        result[j] = copy.deepcopy(result[j])

            self._monitor_update_of_sample(result, self._timer.value, cur_learner_iter)
            return result

    def append(self, ori_data: Any, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Append a data item into queue.
            Add two keys in data:

                - replay_unique_id: The data item's unique id, using ``self._generate_id`` to generate it.
                - replay_buffer_idx: The data item's position index in the queue, this position may already have an \
                    old element, then it would be replaced by this new input one. using ``self._tail`` to locate.
        Arguments:
            - ori_data (:obj:`Any`): The data which will be inserted.
        """
        with self._lock:
            with self._timer:
                if self._deepcopy:
                    data = copy.deepcopy(ori_data)
                else:
                    data = ori_data
                try:
                    assert all([fn(data) for fn in self.check_list])
                except AssertionError:
                    # If data check fails, log it and return without any operations.
                    self._logger.info('Illegal data type [{}], reject it...'.format(type(data)))
                    return
                self._push_count += 1
                # remove->set weight->set data
                if self._data[self._tail] is not None:
                    self._head = (self._tail + 1) % self._replay_buffer_size
                self._remove(self._tail)
                data['replay_unique_id'] = self._generate_id(self._next_unique_id)
                data['replay_buffer_idx'] = self._tail
                self._set_weight(data)
                self._data[self._tail] = data
                self._valid_count += 1
                self._tail = (self._tail + 1) % self._replay_buffer_size
                self._next_unique_id += 1

            self._monitor_update_of_push(1, self._timer.value, cur_collector_envstep)

    def _track_used_data(self, old: Any) -> None:
        if not self._enable_track_used_data:
            return
        if old is not None:
            if isinstance(old, dict) and 'data_id' in old:
                if old['data_id'] not in self._using_data:
                    self._used_data.put(old['data_id'])
                else:
                    self._using_used_data: set
                    self._using_used_data.add(old['data_id'])

    def extend(self, ori_data: List[Any], cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Extend a data list into queue.
            Add two keys in each data item, you can refer to ``append`` for details.
        Arguments:
            - ori_data (:obj:`List[Any]`): The data list.
        """
        with self._lock:
            with self._timer:
                if self._deepcopy:
                    data = copy.deepcopy(ori_data)
                else:
                    data = ori_data
                check_result = [self._data_check(d) for d in data]
                # Only keep data items that pass ``_data_check`.
                valid_data = [d for d, flag in zip(data, check_result) if flag]
                length = len(valid_data)
                # When updating ``_data`` and ``_use_count``, should consider two cases regarding
                # the relationship between "tail + data length" and "queue max length" to check whether
                # data will exceed beyond queue's max length limitation.
                if self._tail + length <= self._replay_buffer_size:
                    for j in range(self._tail, self._tail + length):
                        if self._data[j] is not None:
                            self._head = (j + 1) % self._replay_buffer_size
                        self._remove(j)
                    for i in range(length):
                        valid_data[i]['replay_unique_id'] = self._generate_id(self._next_unique_id + i)
                        valid_data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                        self._set_weight(valid_data[i])
                        self._push_count += 1
                    self._data[self._tail:self._tail + length] = valid_data
                else:
                    data_start = self._tail
                    valid_data_start = 0
                    residual_num = len(valid_data)
                    while True:
                        space = self._replay_buffer_size - data_start
                        L = min(space, residual_num)
                        for j in range(data_start, data_start + L):
                            if self._data[j] is not None:
                                self._head = (j + 1) % self._replay_buffer_size
                            self._remove(j)
                        for i in range(valid_data_start, valid_data_start + L):
                            valid_data[i]['replay_unique_id'] = self._generate_id(self._next_unique_id + i)
                            valid_data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                            self._set_weight(valid_data[i])
                            self._push_count += 1
                        self._data[data_start:data_start + L] = valid_data[valid_data_start:valid_data_start + L]
                        residual_num -= L
                        if residual_num <= 0:
                            break
                        else:
                            data_start = 0
                            valid_data_start += L
                self._valid_count += len(valid_data)
                # Update ``tail`` and ``next_unique_id`` after the whole list is pushed into buffer.
                self._tail = (self._tail + length) % self._replay_buffer_size
                self._next_unique_id += length

            self._monitor_update_of_push(length, self._timer.value, cur_collector_envstep)

    def update(self, info: dict) -> None:
        r"""
        Overview:
            Update a data's priority. Use "repaly_buffer_idx" to locate and "replay_unique_id" to verify.
        Arguments:
            - info (:obj:`dict`): Info dict containing all necessary keys for priority update.
        """
        with self._lock:
            if self._enable_track_used_data:
                used_id = info.get('used_id', [])
                self._using_data.difference_update(used_id)
                for data_id in used_id:
                    if data_id in self._using_used_data:
                        self._using_used_data.remove(data_id)
                        self._used_data.put(data_id)
            if 'priority' not in info:
                return
            data = [info['replay_unique_id'], info['replay_buffer_idx'], info['priority']]
            for id_, idx, priority in zip(*data):
                # Only if the data still exists in the queue, will the update operation be done.
                if self._data[idx] is not None \
                        and self._data[idx]['replay_unique_id'] == id_:  # Verify the same transition(data)
                    assert priority >= 0, priority
                    assert self._data[idx]['replay_buffer_idx'] == idx
                    self._data[idx]['priority'] = priority + self._eps  # Add epsilon to avoid priority == 0
                    self._set_weight(self._data[idx])
                    # Update max priority
                    self._max_priority = max(self._max_priority, priority)

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variables.
        """
        with self._lock:
            for i in range(len(self._data)):
                self._remove(i)
            self._valid_count = 0
            self._head = 0
            self._tail = 0
            self._max_priority = 1.0

    def close(self) -> None:
        """
        Overview:
            Close the tensorboard logger.
        """
        self.clear()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Call ``close`` to delete the object.
        """
        self.close()

    def _set_weight(self, data: Dict) -> None:
        r"""
        Overview:
            Set sumtree and mintree's weight of the input data according to its priority.
            If input data does not have key "priority", it would set to ``self._max_priority`` instead.
        Arguments:
            - data (:obj:`Dict`): The data whose priority(weight) in segement tree should be set/updated.
        """
        if 'priority' not in data.keys() or data['priority'] is None:
            data['priority'] = self._max_priority
        weight = data['priority'] ** self.alpha
        idx = data['replay_buffer_idx']
        self._sum_tree[idx] = weight
        self._min_tree[idx] = weight

    def _data_check(self, d: Any) -> bool:
        r"""
        Overview:
            Data legality check, using rules(functions) in ``self.check_list``.
        Arguments:
            - d (:obj:`Any`): The data which needs to be checked.
        Returns:
            - result (:obj:`bool`): Whether the data passes the check.
        """
        # only the data passes all the check functions, would the check return True
        return all([fn(d) for fn in self.check_list])

    def _get_indices(self, size: int) -> list:
        r"""
        Overview:
            Get the sample index list according to the priority probability.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled
        Returns:
            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.
        """
        # Divide [0, 1) into size intervals on average
        intervals = np.array([i * 1.0 / size for i in range(size)])
        # uniformly sample within each interval
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        # rescale to [0, S), where S is the sum of all datas' priority (root value of sum tree)
        mass *= self._sum_tree.reduce()
        # find prefix sum index to sample with probability
        return [self._sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int) -> None:
        r"""
        Overview:
            Remove a data(set the element in the list to ``None``) and
            update corresponding variables, e.g. sum_tree, min_tree, valid_count.
        Arguments:
            - idx (:obj:`int`): Data at this position will be removed.
        """
        self._track_used_data(self._data[idx])
        if self._data[idx] is not None:
            self._valid_count -= 1
        self._data[idx] = None
        self._sum_tree[idx] = self._sum_tree.neutral_element
        self._min_tree[idx] = self._min_tree.neutral_element
        self._use_count[idx] = 0

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int) -> list:
        r"""
        Overview:
            Sample data with ``indices``; Remove a data item if it is used for too many times.
        Arguments:
            - indices (:obj:`List[int]`): A list including all the sample indices.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - data (:obj:`list`) Sampled data.
        """
        # Calculate max weight for normalizing IS
        sum_tree_root = self._sum_tree.reduce()
        p_min = self._min_tree.reduce() / sum_tree_root
        max_weight = (self._valid_count * p_min) ** (-self._beta)
        data = []
        for idx in indices:
            assert self._data[idx] is not None
            assert self._data[idx]['replay_buffer_idx'] == idx, (self._data[idx]['replay_buffer_idx'], idx)
            if self._deepcopy:
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            if self._enable_track_used_data:
                self._using_data.add(copy_data['data_id'])
            # Store staleness, use and IS(importance sampling weight for gradient step) for monitor and outer use
            self._use_count[idx] += 1
            copy_data['staleness'] = self._calculate_staleness(idx, cur_learner_iter)
            copy_data['use'] = self._use_count[idx]
            p_sample = self._sum_tree[idx] / sum_tree_root
            weight = (self._valid_count * p_sample) ** (-self._beta)
            copy_data['IS'] = weight / max_weight
            data.append(copy_data)
        # Remove datas whose "use count" is greater than ``max_use``
        for idx in indices:
            if self._use_count[idx] >= self._max_use:
                self._remove(idx)
        # Beta annealing
        self._beta = min(1.0, self._beta + self._beta_anneal_one_step)
        return data

    def _monitor_update_of_push(self, add_count: int, add_time: float, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Update values in monitor, then update text logger and tensorboard logger.
            Called in ``append`` and ``extend``.
        Arguments:
            - add_count (:obj:`int`): How many datas are added into buffer.
            - add_time (:obj:`float`): How long does it take to add in such datas.
        """
        self._cur_collector_envstep = cur_collector_envstep
        self._natural_monitor.in_count = add_count
        self._in_tick_monitor.in_time = add_time
        self._in_tick_monitor.time.step()
        in_dict = {
            'in_count_avg': self._natural_monitor.avg['in_count'](),
            'in_time_avg': self._in_tick_monitor.avg['in_time']()
        }
        if self._in_count % self._log_freq == 0:
            self._logger.debug("===Add In Buffer {} Times===".format(self._in_count))
            self._logger.print_vars(in_dict)
            for k, v in in_dict.items():
                iter_metric = self._cur_learner_iter if self._cur_learner_iter != -1 else self._in_count
                step_metric = self._cur_collector_envstep if self._cur_collector_envstep != -1 else self._in_count
                self._tb_logger.add_scalar('buffer_{}_iter/'.format(self.name) + k, v, iter_metric)
                self._tb_logger.add_scalar('buffer_{}_step/'.format(self.name) + k, v, step_metric)
        self._in_count += 1

    def _monitor_update_of_sample(self, sample_data: list, sample_time: float, cur_learner_iter: int) -> None:
        r"""
        Overview:
            Update values in monitor, then update text logger and tensorboard logger.
            Called in ``sample``.
        Arguments:
            - sample_data (:obj:`list`): Sampled data. Used to get sample length and data's attributes, \
                e.g. use, priority, staleness, etc.
            - sample_time (:obj:`float`): How long does it take to sample such datas.
        """
        self._cur_learner_iter = cur_learner_iter
        self._natural_monitor.out_count = len(sample_data)
        self._out_tick_monitor.out_time = sample_time
        use = sum([d['use'] for d in sample_data]) / len(sample_data)
        priority = sum([d['priority'] for d in sample_data]) / len(sample_data)
        staleness = sum([d['staleness'] for d in sample_data]) / len(sample_data)
        self._out_tick_monitor.use = use
        self._out_tick_monitor.priority = priority
        self._out_tick_monitor.staleness = staleness
        self._out_tick_monitor.time.step()
        out_dict = {
            'out_count_avg': self._natural_monitor.avg['out_count'](),
            'out_time_avg': self._out_tick_monitor.avg['out_time'](),
            'use_avg': self._out_tick_monitor.avg['use'](),
            'use_max': self._out_tick_monitor.max['use'](),
            'priority_avg': self._out_tick_monitor.avg['priority'](),
            'priority_max': self._out_tick_monitor.max['priority'](),
            'priority_min': self._out_tick_monitor.min['priority'](),
            'staleness_avg': self._out_tick_monitor.avg['staleness'](),
            'staleness_max': self._out_tick_monitor.max['staleness'](),
        }
        if self._out_count % self._log_freq == 0:
            self._logger.debug("===Read Buffer {} Times===".format(self._out_count))
            self._logger.print_vars(out_dict)
            for k, v in out_dict.items():
                iter_metric = self._cur_learner_iter if self._cur_learner_iter != -1 else self._in_count
                step_metric = self._cur_collector_envstep if self._cur_collector_envstep != -1 else self._in_count
                self._tb_logger.add_scalar('buffer_{}_iter/'.format(self.name) + k, v, iter_metric)
                self._tb_logger.add_scalar('buffer_{}_step/'.format(self.name) + k, v, step_metric)
        self._out_count += 1

    def _calculate_staleness(self, pos_index: int, cur_learner_iter: int) -> Optional[int]:
        r"""
        Overview:
            Calculate a data's staleness according to its own attribute ``collect_iter``
            and input parameter ``cur_learner_iter``.
        Arguments:
            - pos_index (:obj:`int`): The position index. Staleness of the data at this index will be calculated.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - staleness (:obj:`int`): Staleness of data at position ``pos_index``.

        .. note::
            Caller should guarantee that data at ``pos_index`` is not None; Otherwise this function may raise an error.
        """
        if self._data[pos_index] is None:
            raise ValueError("Prioritized's data at index {} is None".format(pos_index))
        else:
            # Calculate staleness, remove it if too stale
            collect_iter = self._data[pos_index].get('collect_iter', cur_learner_iter + 1)
            if isinstance(collect_iter, list):
                # Timestep transition's collect_iter is a list
                collect_iter = min(collect_iter)
            # ``staleness`` might be -1, means invalid, e.g. collector does not report collecting model iter,
            # or it is a demonstration buffer(which means data is not generated by collector) etc.
            staleness = cur_learner_iter - collect_iter
            return staleness

    def _generate_id(self, data_id: int) -> str:
        """
        Overview:
            Use ``self.name`` and input ``id`` to generate a unique id for next data to be inserted.
        Arguments:
            - data_id (:obj:`int`): Current unique id.
        Returns:
            - id (:obj:`str`): Id in format "BufferName_DataId".
        """
        return "{}_{}".format(self.name, str(data_id))

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

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
            if not self._used_data.empty():
                return self._used_data.get()
            else:
                return None
        else:
            return None

    @property
    def push_count(self) -> int:
        return self._push_count

    @property
    def replay_start_size(self) -> int:
        return self._replay_start_size

    def state_dict(self) -> dict:
        return {
            'data': self._data,
            'use_count': self._use_count,
            'tail': self._tail,
            'max_priority': self._max_priority,
            'anneal_step': self._anneal_step,
            'beta': self._beta,
            'head': self._head,
            'next_unique_id': self._next_unique_id,
            'valid_count': self._valid_count,
            'sum_tree': self._sum_tree,
            'min_tree': self._min_tree,
        }

    def load_state_dict(self, _state_dict: dict) -> None:
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self.extend(_state_dict['data'])
        else:
            for k, v in _state_dict.items():
                setattr(self, '_{}'.format(k), v)
