import copy
import time
from typing import Union, NoReturn, Any, Optional, List, Dict, Tuple
import numpy as np

from ding.worker.replay_buffer import IBuffer
from ding.utils import SumSegmentTree, MinSegmentTree, BUFFER_REGISTRY
from ding.utils import LockContext, LockContextType, build_logger
from ding.utils.autolog import TickTime
from .utils import UsedDataRemover, generate_id, SampledDataAttrMonitor, PeriodicThruputMonitor, ThruputController


def to_positive_index(idx: Union[int, None], size: int) -> int:
    if idx is None or idx >= 0:
        return idx
    else:
        return size + idx


@BUFFER_REGISTRY.register('advanced')
class AdvancedReplayBuffer(IBuffer):
    r"""
    Overview:
        Prioritized replay buffer derived from ``NaiveReplayBuffer``.
        This replay buffer adds:

            1) Prioritized experience replay implemented by segment tree.
            2) Data quality monitor. Monitor use count and staleness of each data.
            3) Throughput monitor and control.
            4) Logger. Log 2) and 3) in tensorboard or text.
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    Property:
        beta, replay_buffer_size, push_count
    """

    config = dict(
        type='advanced',
        # Max length of the buffer.
        replay_buffer_size=4096,
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
        # Whether to track the used data. Used data means they are removed out of buffer and would never be used again.
        enable_track_used_data=False,
        # Whether to deepcopy data when willing to insert and sample data. For security purpose.
        deepcopy=False,
        thruput_controller=dict(
            # Rate limit. The ratio of "Sample Count" to "Push Count" should be in [min, max] range.
            # If greater than max ratio, return `None` when calling ``sample```;
            # If smaller than min ratio, throw away the new data when calling ``push``.
            push_sample_rate_limit=dict(
                max=float("inf"),
                min=0,
            ),
            # Controller will take how many seconds into account, i.e. For the past `window_seconds` seconds,
            # sample_push_rate will be calculated and campared with `push_sample_rate_limit`.
            window_seconds=30,
            # The minimum ratio that buffer must satisfy before anything can be sampled.
            # The ratio is calculated by "Valid Count" divided by "Batch Size".
            # E.g. sample_min_limit_ratio = 2.0, valid_count = 50, batch_size = 32, it is forbidden to sample.
            sample_min_limit_ratio=1,
        ),
        # Monitor configuration for monitor and logger to use. This part does not affect buffer's function.
        monitor=dict(
            sampled_data_attr=dict(
                # Past datas will be used for moving average.
                average_range=5,
                # Print data attributes every `print_freq` samples.
                print_freq=200,  # times
            ),
            periodic_thruput=dict(
                # Every `seconds` seconds, thruput(push/sample/remove count) will be printed.
                seconds=60,
            ),
        ),
    )

    def __init__(
            self,
            cfg: dict,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'buffer',
    ) -> int:
        """
        Overview:
            Initialize the buffer
        Arguments:
            - cfg (:obj:`dict`): Config dict.
            - tb_logger (:obj:`Optional['SummaryWriter']`): Outer tb logger. Usually get this argument in serial mode.
            - exp_name (:obj:`Optional[str]`): Name of this experiment.
            - instance_name (:obj:`Optional[str]`): Name of this instance.
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._end_flag = False
        self._cfg = cfg
        self._replay_buffer_size = self._cfg.replay_buffer_size
        self._deepcopy = self._cfg.deepcopy
        # ``_data`` is a circular queue to store data (full data or meta data)
        self._data = [None for _ in range(self._replay_buffer_size)]
        # Current valid data count, indicating how many elements in ``self._data`` is valid.
        self._valid_count = 0
        # How many pieces of data have been pushed into this buffer, should be no less than ``_valid_count``.
        self._push_count = 0
        # Point to the tail position where next data can be inserted, i.e. latest inserted data's next position.
        self._tail = 0
        # Is used to generate a unique id for each data: If a new data is inserted, its unique id will be this.
        self._next_unique_id = 0
        # Lock to guarantee thread safe
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        # Point to the head of the circular queue. The true data is the stalest(oldest) data in this queue.
        # Because buffer would remove data due to staleness or use count, and at the beginning when queue is not
        # filled with data head would always be 0, so ``head`` may be not equal to ``tail``;
        # Otherwise, they two should be the same. Head is used to optimize staleness check in ``_sample_check``.
        self._head = 0
        # use_count is {position_idx: use_count}
        self._use_count = {idx: 0 for idx in range(self._cfg.replay_buffer_size)}
        # Max priority till now. Is used to initizalize a data's priority if "priority" is not passed in with the data.
        self._max_priority = 1.0
        # A small positive number to avoid edge-case, e.g. "priority" == 0.
        self._eps = 1e-5
        # Data check function list, used in ``_append`` and ``_extend``. This buffer requires data to be dict.
        self.check_list = [lambda x: isinstance(x, dict)]

        self._max_use = self._cfg.max_use
        self._max_staleness = self._cfg.max_staleness
        self.alpha = self._cfg.alpha
        assert 0 <= self.alpha <= 1, self.alpha
        self._beta = self._cfg.beta
        assert 0 <= self._beta <= 1, self._beta
        self._anneal_step = self._cfg.anneal_step
        if self._anneal_step != 0:
            self._beta_anneal_step = (1 - self._beta) / self._anneal_step

        # Prioritized sample.
        # Capacity needs to be the power of 2.
        capacity = int(np.power(2, np.ceil(np.log2(self.replay_buffer_size))))
        # Sum segtree and min segtree are used to sample data according to priority.
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)

        # Thruput controller
        push_sample_rate_limit = self._cfg.thruput_controller.push_sample_rate_limit
        self._always_can_push = True if push_sample_rate_limit['max'] == float('inf') else False
        self._always_can_sample = True if push_sample_rate_limit['min'] == 0 else False
        self._use_thruput_controller = not self._always_can_push or not self._always_can_sample
        if self._use_thruput_controller:
            self._thruput_controller = ThruputController(self._cfg.thruput_controller)
        self._sample_min_limit_ratio = self._cfg.thruput_controller.sample_min_limit_ratio
        assert self._sample_min_limit_ratio >= 1

        # Monitor & Logger
        monitor_cfg = self._cfg.monitor
        if tb_logger is not None:
            self._logger, _ = build_logger(
                './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                './{}/log/{}'.format(self._exp_name, self._instance_name),
                self._instance_name,
            )
        self._start_time = time.time()
        # Sampled data attributes.
        self._cur_learner_iter = -1
        self._cur_collector_envstep = -1
        self._sampled_data_attr_print_count = 0
        self._sampled_data_attr_monitor = SampledDataAttrMonitor(
            TickTime(), expire=monitor_cfg.sampled_data_attr.average_range
        )
        self._sampled_data_attr_print_freq = monitor_cfg.sampled_data_attr.print_freq
        # Periodic thruput.
        self._periodic_thruput_monitor = PeriodicThruputMonitor(
            self._instance_name, monitor_cfg.periodic_thruput, self._logger, self._tb_logger
        )

        # Used data remover
        self._enable_track_used_data = self._cfg.enable_track_used_data
        if self._enable_track_used_data:
            self._used_data_remover = UsedDataRemover()

    def start(self) -> None:
        """
        Overview:
            Start the buffer's used_data_remover thread if enables track_used_data.
        """
        if self._enable_track_used_data:
            self._used_data_remover.start()

    def close(self) -> None:
        """
        Overview:
            Clear the buffer; Join the buffer's used_data_remover thread if enables track_used_data.
            Join periodic throughtput monitor, flush tensorboard logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self.clear()
        self._periodic_thruput_monitor.close()
        self._tb_logger.flush()
        self._tb_logger.close()
        if self._enable_track_used_data:
            self._used_data_remover.close()

    def sample(self, size: int, cur_learner_iter: int, sample_range: slice = None) -> Optional[list]:
        """
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which \
                means only sample among the last 10 data
        Returns:
            - sample_data (:obj:`list`): A list of data with length ``size``
        ReturnsKeys:
            - necessary: original keys(e.g. `obs`, `action`, `next_obs`, `reward`, `info`), \
                `replay_unique_id`, `replay_buffer_idx`
            - optional(if use priority): `IS`, `priority`
        """
        if size == 0:
            return []
        can_sample_stalenss, staleness_info = self._sample_check(size, cur_learner_iter)
        if self._always_can_sample:
            can_sample_thruput, thruput_info = True, "Always can sample because push_sample_rate_limit['min'] == 0"
        else:
            can_sample_thruput, thruput_info = self._thruput_controller.can_sample(size)
        if not can_sample_stalenss or not can_sample_thruput:
            self._logger.info(
                'Refuse to sample due to -- \nstaleness: {}, {} \nthruput: {}, {}'.format(
                    not can_sample_stalenss, staleness_info, not can_sample_thruput, thruput_info
                )
            )
            return None
        with self._lock:
            indices = self._get_indices(size, sample_range)
            result = self._sample_with_indices(indices, cur_learner_iter)
            # Deepcopy ``result``'s same indice datas in case ``self._get_indices`` may get datas with
            # the same indices, i.e. the same datas would be sampled afterwards.
            # if self._deepcopy==True -> all data is different
            # if len(indices) == len(set(indices)) -> no duplicate data
            if not self._deepcopy and len(indices) != len(set(indices)):
                for i, index in enumerate(indices):
                    tmp = []
                    for j in range(i + 1, size):
                        if index == indices[j]:
                            tmp.append(j)
                    for j in tmp:
                        result[j] = copy.deepcopy(result[j])
            self._monitor_update_of_sample(result, cur_learner_iter)
            return result

    def push(self, data: Union[List[Any], Any], cur_collector_envstep: int) -> None:
        r"""
        Overview:
            Push a data into buffer.
        Arguments:
            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \
                (in `Any` type), or many(int `List[Any]` type).
            - cur_collector_envstep (:obj:`int`): Collector's current env step.
        """
        push_size = len(data) if isinstance(data, list) else 1
        if self._always_can_push:
            can_push, push_info = True, "Always can push because push_sample_rate_limit['max'] == float('inf')"
        else:
            can_push, push_info = self._thruput_controller.can_push(push_size)
        if not can_push:
            self._logger.info('Refuse to push because {}'.format(push_info))
            return
        if isinstance(data, list):
            self._extend(data, cur_collector_envstep)
        else:
            self._append(data, cur_collector_envstep)

    def _sample_check(self, size: int, cur_learner_iter: int) -> Tuple[bool, str]:
        r"""
        Overview:
            Do preparations for sampling and check whether data is enough for sampling
            Preparation includes removing stale datas in ``self._data``.
            Check includes judging whether this buffer has more than ``size`` datas to sample.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.
        Returns:
            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.
            - str_info (:obj:`str`): Str type info, explaining why cannot sample. (If can sample, return "Can sample")

        .. note::
            This function must be called before data sample.
        """
        staleness_remove_count = 0
        with self._lock:
            if self._max_staleness != float("inf"):
                p = self._head
                while True:
                    if self._data[p] is not None:
                        staleness = self._calculate_staleness(p, cur_learner_iter)
                        if staleness >= self._max_staleness:
                            self._remove(p)
                            staleness_remove_count += 1
                        else:
                            # Since the circular queue ``self._data`` guarantees that data's staleness is decreasing
                            # from index self._head to index self._tail - 1, we can jump out of the loop as soon as
                            # meeting a fresh enough data
                            break
                    p = (p + 1) % self._replay_buffer_size
                    if p == self._tail:
                        # Traverse a circle and go back to the tail, which means can stop staleness checking now
                        break
            str_info = "Remove {} elements due to staleness. ".format(staleness_remove_count)
            if self._valid_count / size < self._sample_min_limit_ratio:
                str_info += "Not enough for sampling. valid({}) / sample({}) < sample_min_limit_ratio({})".format(
                    self._valid_count, size, self._sample_min_limit_ratio
                )
                return False, str_info
            else:
                str_info += "Can sample."
                return True, str_info

    def _append(self, ori_data: Any, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Append a data item into queue.
            Add two keys in data:

                - replay_unique_id: The data item's unique id, using ``generate_id`` to generate it.
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
            data['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id)
            data['replay_buffer_idx'] = self._tail
            self._set_weight(data)
            self._data[self._tail] = data
            self._valid_count += 1
            self._periodic_thruput_monitor.valid_count = self._valid_count
            self._tail = (self._tail + 1) % self._replay_buffer_size
            self._next_unique_id += 1
            self._monitor_update_of_push(1, cur_collector_envstep)

    def _extend(self, ori_data: List[Any], cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Extend a data list into queue.
            Add two keys in each data item, you can refer to ``_append`` for more details.
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
                    valid_data[i]['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id + i)
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
                        valid_data[i]['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id + i)
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
            self._periodic_thruput_monitor.valid_count = self._valid_count
            # Update ``tail`` and ``next_unique_id`` after the whole list is pushed into buffer.
            self._tail = (self._tail + length) % self._replay_buffer_size
            self._next_unique_id += length
            self._monitor_update_of_push(length, cur_collector_envstep)

    def update(self, info: dict) -> None:
        r"""
        Overview:
            Update a data's priority. Use `repaly_buffer_idx` to locate, and use `replay_unique_id` to verify.
        Arguments:
            - info (:obj:`dict`): Info dict containing all necessary keys for priority update.
        ArgumentsKeys:
            - necessary: `replay_unique_id`, `replay_buffer_idx`, `priority`. All values are lists with the same length.
        """
        with self._lock:
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
                else:
                    self._logger.debug(
                        '[Skip Update]: buffer_idx: {}; id_in_buffer: {}; id_in_update_info: {}'.format(
                            idx, id_, priority
                        )
                    )

    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variables.
        """
        with self._lock:
            for i in range(len(self._data)):
                self._remove(i)
            assert self._valid_count == 0, self._valid_count
            self._head = 0
            self._tail = 0
            self._max_priority = 1.0

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

    def _get_indices(self, size: int, sample_range: slice = None) -> list:
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
        if sample_range is None:
            # Rescale to [0, S), where S is the sum of all datas' priority (root value of sum tree)
            mass *= self._sum_tree.reduce()
        else:
            # Rescale to [a, b)
            start = to_positive_index(sample_range.start, self._replay_buffer_size)
            end = to_positive_index(sample_range.stop, self._replay_buffer_size)
            a = self._sum_tree.reduce(0, start)
            b = self._sum_tree.reduce(0, end)
            mass = mass * (b - a) + a
        # Find prefix sum index to sample with probability
        return [self._sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int, use_too_many_times: bool = False) -> None:
        r"""
        Overview:
            Remove a data(set the element in the list to ``None``) and update corresponding variables,
            e.g. sum_tree, min_tree, valid_count.
        Arguments:
            - idx (:obj:`int`): Data at this position will be removed.
        """
        if use_too_many_times:
            if self._enable_track_used_data:
                # Must track this data, but in parallel mode.
                # Do not remove it, but make sure it will not be sampled.
                self._data[idx]['priority'] = 0
                self._sum_tree[idx] = self._sum_tree.neutral_element
                self._min_tree[idx] = self._min_tree.neutral_element
                return
            elif idx == self._head:
                # Correct `self._head` when the queue head is removed due to use_count
                self._head = (self._head + 1) % self._replay_buffer_size
        if self._data[idx] is not None:
            if self._enable_track_used_data:
                self._used_data_remover.add_used_data(self._data[idx])
            self._valid_count -= 1
            self._periodic_thruput_monitor.valid_count = self._valid_count
            self._periodic_thruput_monitor.remove_data_count += 1
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
            # Store staleness, use and IS(importance sampling weight for gradient step) for monitor and outer use
            self._use_count[idx] += 1
            copy_data['staleness'] = self._calculate_staleness(idx, cur_learner_iter)
            copy_data['use'] = self._use_count[idx]
            p_sample = self._sum_tree[idx] / sum_tree_root
            weight = (self._valid_count * p_sample) ** (-self._beta)
            copy_data['IS'] = weight / max_weight
            data.append(copy_data)
        if self._max_use != float("inf"):
            # Remove datas whose "use count" is greater than ``max_use``
            for idx in indices:
                if self._use_count[idx] >= self._max_use:
                    self._remove(idx, use_too_many_times=True)
        # Beta annealing
        if self._anneal_step != 0:
            self._beta = min(1.0, self._beta + self._beta_anneal_step)
        return data

    def _monitor_update_of_push(self, add_count: int, cur_collector_envstep: int = -1) -> None:
        r"""
        Overview:
            Update values in monitor, then update text logger and tensorboard logger.
            Called in ``_append`` and ``_extend``.
        Arguments:
            - add_count (:obj:`int`): How many datas are added into buffer.
            - cur_collector_envstep (:obj:`int`): Collector envstep, passed in by collector.
        """
        self._periodic_thruput_monitor.push_data_count += add_count
        if self._use_thruput_controller:
            self._thruput_controller.history_push_count += add_count
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
        self._periodic_thruput_monitor.sample_data_count += len(sample_data)
        if self._use_thruput_controller:
            self._thruput_controller.history_sample_count += len(sample_data)
        self._cur_learner_iter = cur_learner_iter
        use_avg = sum([d['use'] for d in sample_data]) / len(sample_data)
        use_max = max([d['use'] for d in sample_data])
        priority_avg = sum([d['priority'] for d in sample_data]) / len(sample_data)
        priority_max = max([d['priority'] for d in sample_data])
        priority_min = min([d['priority'] for d in sample_data])
        staleness_avg = sum([d['staleness'] for d in sample_data]) / len(sample_data)
        staleness_max = max([d['staleness'] for d in sample_data])
        self._sampled_data_attr_monitor.use_avg = use_avg
        self._sampled_data_attr_monitor.use_max = use_max
        self._sampled_data_attr_monitor.priority_avg = priority_avg
        self._sampled_data_attr_monitor.priority_max = priority_max
        self._sampled_data_attr_monitor.priority_min = priority_min
        self._sampled_data_attr_monitor.staleness_avg = staleness_avg
        self._sampled_data_attr_monitor.staleness_max = staleness_max
        self._sampled_data_attr_monitor.time.step()
        out_dict = {
            'use_avg': self._sampled_data_attr_monitor.avg['use'](),
            'use_max': self._sampled_data_attr_monitor.max['use'](),
            'priority_avg': self._sampled_data_attr_monitor.avg['priority'](),
            'priority_max': self._sampled_data_attr_monitor.max['priority'](),
            'priority_min': self._sampled_data_attr_monitor.min['priority'](),
            'staleness_avg': self._sampled_data_attr_monitor.avg['staleness'](),
            'staleness_max': self._sampled_data_attr_monitor.max['staleness'](),
            'beta': self._beta,
        }
        if self._sampled_data_attr_print_count % self._sampled_data_attr_print_freq == 0:
            self._logger.info("=== Sample data {} Times ===".format(self._sampled_data_attr_print_count))
            self._logger.info(self._logger.get_tabulate_vars_hor(out_dict))
            for k, v in out_dict.items():
                iter_metric = self._cur_learner_iter if self._cur_learner_iter != -1 else None
                step_metric = self._cur_collector_envstep if self._cur_collector_envstep != -1 else None
                if iter_metric is not None:
                    self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, iter_metric)
                if step_metric is not None:
                    self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, step_metric)
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

    def count(self) -> int:
        """
        Overview:
            Count how many valid datas there are in the buffer.
        Returns:
            - count (:obj:`int`): Number of valid data.
        """
        return self._valid_count

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> NoReturn:
        self._beta = beta

    def state_dict(self) -> dict:
        """
        Overview:
            Provide a state dict to keep a record of current buffer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer. \
                With the dict, one can easily reproduce the buffer.
        """
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

    def load_state_dict(self, _state_dict: dict, deepcopy: bool = False) -> None:
        """
        Overview:
            Load state dict to reproduce the buffer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.
        """
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self._extend(_state_dict['data'])
        else:
            for k, v in _state_dict.items():
                if deepcopy:
                    setattr(self, '_{}'.format(k), copy.deepcopy(v))
                else:
                    setattr(self, '_{}'.format(k), v)

    @property
    def replay_buffer_size(self) -> int:
        return self._replay_buffer_size

    @property
    def push_count(self) -> int:
        return self._push_count
