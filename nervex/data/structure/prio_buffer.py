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
import threading

from nervex.data.structure.naive_buffer import NaiveReplayBuffer
from nervex.data.structure.segment_tree import SumSegmentTree, MinSegmentTree
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from nervex.utils import LockContext, LockContextType, EasyTimer, build_logger, deep_merge_dicts


class PrioritizedReplayBuffer(NaiveReplayBuffer):
    r"""
    Overview:
        Prioritized replay buffer derived from ``NaiveReplayBuffer``.
        This replay buffer adds:
            1) Prioritized experience replay implemented through segment tree.
            2) Use count and staleness of each data, to guarantee data quality.
            3) Monitor mechanism to watch in-and-out data flow attributes.
    Interface:
        __init__, append, extend, sample, update, clear, close
    Property:
        replay_buffer_size, validlen, beta
    """

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        type='priority',
        # Max length of the buffer.
        replay_buffer_size=4096,
        # start training data count
        replay_buffer_start_size=0,
        # Max use times of one data in the buffer. Data will be removed once used for too many times.
        max_use=float("inf"),
        # Max staleness time duration of one data in the buffer; Data will be removed if
        # the duration from collecting to training is too long, i.e. The data is too stale.
        max_staleness=float("inf"),
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        beta=0.4,
        # Anneal step for beta: 0 means no annealing
        anneal_step=int(1e5),
        # Whether to track the used data or not. Buffer will use a new data structure to track data if set True.
        enable_track_used_data=False,
        # Whether to deepcopy data when willing to insert and sample data. For security purpose.
        deepcopy=False,
        # Monitor configuration for monitor and logger to use. This part does not affect buffer's function.
        monitor=dict(
            # Logger's save path
            log_path='./log/buffer/',
            sampled_data_attr=dict(
                # Past datas will be used for moving average.
                average_range=5,
                # Print data attributes every `print_freq` samples.
                print_freq=200,  # times
            ),
            periodic_thruput=dict(
                # Every `seconds` seconds, thruput(push/sample/remove count) will be printed.
                seconds=60,  # seconds 
            ),
        ),
    )

    def __init__(
            self,
            name: str,
            cfg: dict,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
    ) -> int:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - name (:obj:`str`): Buffer name, mainly used to generate unique data id and logger name.
        """
        self._cfg = cfg
        # ``_data`` is a circular queue to store data (or data's reference/file path)
        self._data = [None for _ in range(self._cfg.replay_buffer_size)]
        self._enable_track_used_data = self._cfg.enable_track_used_data
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
        self._use_count = {idx: 0 for idx in range(self._cfg.replay_buffer_size)}
        # Max priority till now. Is used to initizalize a data's priority if "priority" is not passed in with the data.
        self._max_priority = 1.0
        # A small positive number to avoid edge-case, e.g. "priority" == 0.
        self._eps = 1e-3
        # Data check function list, used in ``append`` and ``extend``. This buffer requires data to be dict.
        self.check_list = [lambda x: isinstance(x, dict)]
        # Lock to guarantee thread safe
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)

        self.name = name
        self._replay_buffer_size = self._cfg.replay_buffer_size
        self._replay_buffer_start_size = self._cfg.replay_buffer_start_size
        self._max_use = self._cfg.max_use
        self._max_staleness = self._cfg.max_staleness
        self.alpha = self._cfg.alpha
        assert 0 <= self.alpha <= 1, self.alpha
        self._beta = self._cfg.beta
        assert 0 <= self._beta <= 1, self._beta
        self._anneal_step = self._cfg.anneal_step
        if self._anneal_step != 0:
            self._beta_anneal_step = (1 - self._beta) / self._anneal_step
        self._deepcopy = self._cfg.deepcopy

        self._end_flag = False

        # Prioritized sample.
        # Capacity needs to be the power of 2.
        capacity = int(np.power(2, np.ceil(np.log2(self.replay_buffer_size))))
        # Sum segtree and min segtree are used to sample data according to priority.
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)

        # Monitor & Logger
        monitor_cfg = self._cfg.monitor
        if tb_logger is not None:
            self._logger, _ = build_logger(monitor_cfg.log_path, self.name + '_buffer', need_tb=False)
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                monitor_cfg.log_path,
                self.name + '_buffer',
            )
        # Sampled data attributes.
        self._cur_learner_iter = -1
        self._cur_collector_envstep = -1
        self._sampled_data_attr_print_count = 0
        self._sampled_data_attr_monitor = SampledDataAttrMonitor(TickTime(), expire=monitor_cfg.sampled_data_attr.average_range)
        self._sampled_data_attr_print_freq = monitor_cfg.sampled_data_attr.print_freq
        # Periodic thruput.
        self._thruput_print_seconds = monitor_cfg.periodic_thruput.seconds
        self._thruput_print_times = 0
        self._thruput_start_time = time.time()
        self._push_data_count = 0
        self._sample_data_count = 0
        self._remove_data_count = 0
        self._thruput_log_thread = threading.Thread(
            target=self._thrput_print_periodically, args=(), name='periodic_thruput_log'
        )
        self._thruput_log_thread.daemon = True
        self._thruput_log_thread.start()


    def sample_check(self, size: int, cur_learner_iter: int) -> bool:
        r"""
        Overview:
            Do preparations for sampling and check whther data is enough for sampling
            Preparation includes removing stale transition in ``self._data``.
            Check includes judging whether this buffer has more than ``size`` datas to sample.
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
                        # index self._head to index self._tail - 1, we can jump out of the loop as soon as
                        # meeting a fresh enough data
                        break
                p = (p + 1) % self._replay_buffer_size
                if p == self._tail:
                    # Traverse a circle and go back to the tail, which means can stop staleness checking now
                    break
            if self._valid_count < size:
                self._logger.info(
                    "No enough elements for sampling (expect: {} / current: {})".format(size, self._valid_count)
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
            - sample_data (:obj:`list`): A list of data with length ``size``; \
                Each data owns keys: original keys + ['IS', 'priority', 'replay_unique_id', 'replay_buffer_idx'].

        .. note::
            Before calling this function, ``sample_check`` must be called.
        """
        if size == 0:
            return []
        with self._lock:
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
            self._monitor_update_of_sample(result, cur_learner_iter)
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
            - cur_collector_envstep (:obj:`int`): Collector's current env step, used to draw tensorboard.
        """
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            try:
                assert self._data_check(data)
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
            self._monitor_update_of_push(1, cur_collector_envstep)

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
            - cur_collector_envstep (:obj:`int`): Collector's current env step, used to draw tensorboard.
        """
        with self._lock:
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
            self._monitor_update_of_push(length, cur_collector_envstep)

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
        self._end_flag = True

    def __del__(self) -> None:
        """
        Overview:
            Call ``close`` to delete the object.
        """
        if not self._end_flag:
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
        # Uniformly sample within each interval
        mass = intervals + np.random.uniform(size=(size, )) * 1. / size
        # Rescale to [0, S), where S is the sum of all datas' priority (root value of sum tree)
        mass *= self._sum_tree.reduce()
        # Find prefix sum index to sample with probability
        return [self._sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int) -> None:
        r"""
        Overview:
            Remove a data(set the element in the list to ``None``) and
            update corresponding variables, e.g. sum_tree, min_tree, valid_count.
        Arguments:
            - idx (:obj:`int`): Data at this position will be removed.
        """
        if idx == self._head:
            self._head = (self._head + 1) % self._replay_buffer_size
        self._track_used_data(self._data[idx])
        if self._data[idx] is not None:
            self._valid_count -= 1
            self._remove_data_count += 1
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
        if self._anneal_step != 0:
            self._beta = min(1.0, self._beta + self._beta_anneal_step)
        return data

    def _monitor_update_of_push(self, add_count: int, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Update values in monitor, then update text logger and tensorboard logger.
            Called in ``append`` and ``extend``.
        Arguments:
            - add_count (:obj:`int`): How many datas are added into buffer.
            - cur_collector_envstep (:obj:`int`): Collector envstep, passed in by collector.
        """
        self._push_data_count += add_count
        self._cur_collector_envstep = cur_collector_envstep

    def _monitor_update_of_sample(self, sample_data: list, cur_learner_iter: int) -> None:
        r"""
        Overview:
            Update values in monitor, then update text logger and tensorboard logger.
            Called in ``sample``.
        Arguments:
            - sample_data (:obj:`list`): Sampled data. Used to get sample length and data's attributes, \
                e.g. use, priority, staleness, etc.
            - cur_learner_iter (:obj:`int`): Learner iteration, passed in by learner.
        """
        self._sample_data_count += len(sample_data)
        self._cur_learner_iter = cur_learner_iter
        use = sum([d['use'] for d in sample_data]) / len(sample_data)
        priority = sum([d['priority'] for d in sample_data]) / len(sample_data)
        staleness = sum([d['staleness'] for d in sample_data]) / len(sample_data)
        self._sampled_data_attr_monitor.use = use
        self._sampled_data_attr_monitor.priority = priority
        self._sampled_data_attr_monitor.staleness = staleness
        self._sampled_data_attr_monitor.time.step()
        out_dict = {
            'use_avg': self._sampled_data_attr_monitor.avg['use'](),
            'use_max': self._sampled_data_attr_monitor.max['use'](),
            'priority_avg': self._sampled_data_attr_monitor.avg['priority'](),
            'priority_max': self._sampled_data_attr_monitor.max['priority'](),
            'priority_min': self._sampled_data_attr_monitor.min['priority'](),
            'staleness_avg': self._sampled_data_attr_monitor.avg['staleness'](),
            'staleness_max': self._sampled_data_attr_monitor.max['staleness'](),
        }
        if self._sampled_data_attr_print_count % self._sampled_data_attr_print_freq == 0:
            self._logger.info("=== Sample data {} Times ===".format(self._sampled_data_attr_print_count))
            self._logger.print_vars(out_dict)
            for k, v in out_dict.items():
                iter_metric = self._cur_learner_iter if self._cur_learner_iter != -1 else None
                step_metric = self._cur_collector_envstep if self._cur_collector_envstep != -1 else None
                if iter_metric is not None:
                    self._tb_logger.add_scalar('buffer_{}_iter/'.format(self.name) + k, v, iter_metric)
                if step_metric is not None:
                    self._tb_logger.add_scalar('buffer_{}_step/'.format(self.name) + k, v, step_metric)
        self._sampled_data_attr_print_count += 1

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
                self._logger.print_vars(count_dict)
                for k, v in count_dict.items():
                    self._tb_logger.add_scalar('buffer_{}_sec/'.format(self.name) + k, v, self._count_print_times)
                self._push_data_count = 0
                self._sample_data_count = 0
                self._remove_data_count = 0
                self._thruput_start_time = time.time()
                self._count_print_times += 1
            else:
                time.sleep(min(1, self._thruput_print_seconds * 0.2))

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
    def replay_buffer_start_size(self) -> int:
        return self._replay_buffer_start_size

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
            'push_count': self._push_count,
            'sum_tree': self._sum_tree,
            'min_tree': self._min_tree,
        }


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
