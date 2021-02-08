"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. Base learner class for both serial and parallel training pipeline.
"""
import os
from typing import Any, Union, Callable, List
from functools import partial
from easydict import EasyDict
import torch
from collections import namedtuple

from nervex.data import AsyncDataLoader, default_collate
from nervex.config import base_learner_default_config
from nervex.torch_utils import build_checkpoint_helper, CountVar, auto_checkpoint, build_log_buffer, to_device
from nervex.utils import build_logger, EasyTimer, pretty_print, get_task_uid, import_module
from nervex.utils import deep_merge_dicts, get_rank
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook


class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info during training.
        Info includes: cur_lr, time(data, train, forward, backward), loss(total,...)
        These info variables are firstly recorded in ``log_buffer``, then in ``LearnerHook`` will vars in
        in this monitor be updated by``log_buffer``, finally printed to ``TextLogger`` and ``TensorBoradLogger``.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    data_time = LoggedValue(float)
    data_preprocess_time = LoggedValue(float)
    train_time = LoggedValue(float)
    total_collect_step = LoggedValue(float)
    total_step = LoggedValue(float)
    total_episode = LoggedValue(float)
    total_sample = LoggedValue(float)
    total_duration = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        def __val_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            return records[-1][1]

        for k in getattr(self, '_LoggedModel__properties'):
            self.register_attribute_value('avg', k, partial(__avg_func, prop_name=k))
            self.register_attribute_value('val', k, partial(__val_func, prop_name=k))


def get_simple_monitor_type(properties: List[str] = []) -> TickMonitor:
    """
    Overview:
        Besides basic training variables provided in ``TickMonitor``, many policies have their own customized
        ones to record and monitor. This function can return a customized tick monitor.
        Compared with ``TickMonitor``, ``SimpleTickMonitor`` can record extra ``properties`` passed in by a policy.
    Argumenst:
         - properties (:obj:`List[str]`): Customized properties to monitor.
    Returns:
        - simple_tick_monitor (:obj:`SimpleTickMonitor`): A simple customized tick monitor.
    """
    if len(properties) == 0:
        return TickMonitor
    else:
        attrs = {}
        properties = [
            'data_time', 'data_preprocess_time', 'train_time', 'total_collect_step', 'total_step', 'total_sample',
            'total_episode', 'total_duration'
        ] + properties
        for p_name in properties:
            attrs[p_name] = LoggedValue(float)
        return type('SimpleTickMonitor', (TickMonitor, ), attrs)


class BaseLearner(object):
    r"""
    Overview:
        Base class for model learning.
    Interface:
        __init__, train, start, setup_dataloader, close, call_hook, register_hook, save_checkpoint
    Property:
        learn_info, priority_info, collect_info, last_iter, name, rank, policy
        tick_time, monitor, log_buffer, logger, tb_logger, load_path, checkpoint_manager
    """

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Init method. Load config and use ``self._cfg`` setting to build common learner components,
            e.g. logger helper, checkpoint helper, hooks.
            Policy is not initialized here, but set afterwards through policy setter.
        Arguments:
            - cfg (:obj:`EasyDict`): Learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
        Notes:
            If you want to debug in sync CUDA mode, please add the following code at the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._cfg = deep_merge_dicts(base_learner_default_config, cfg)
        self._learner_uid = get_task_uid()
        self._load_path = self._cfg.load_path
        self._use_cuda = self._cfg.get('use_cuda', False)
        self._use_distributed = self._cfg.use_distributed

        # Learner rank. Used to discriminate which GPU it uses.
        self._rank = get_rank()
        self._device = 'cuda:{}'.format(self._rank % 8) if self._use_cuda else 'cpu'

        # Logger (Monitor is initialized in policy setter)
        # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
        self._timer = EasyTimer()
        rank0 = True if self._rank == 0 else False
        self._logger, self._tb_logger = build_logger('./log/learner', 'learner', need_tb=rank0)
        self._log_buffer = build_log_buffer()

        # Checkpoint helper. Used to save model checkpoint.
        self._checkpointer_manager = build_checkpoint_helper(self._cfg)
        # Learner hook. Used to do specific things at specific time point. Will be set in ``_setup_hook``
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        # Collect info passed from actor. Used in serial entry for message passing.
        self._collect_info = {}
        # Priority info. Used to update replay buffer according to data's priority.
        self._priority_info = None
        # Last iteration. Used to record current iter.
        self._last_iter = CountVar(init_val=0)
        self.info(pretty_print({
            "config": self._cfg,
        }, direct_print=False))
        self._end_flag = False

        # Setup wrapper and hook.
        self._setup_wrapper()
        self._setup_hook()

    def _setup_hook(self) -> None:
        """
        Overview:
            Setup hook for base_learner. Hook is the way to implement some functions at specific time point
            in base_learner. You can refer to ``learner_hook.py``.
        """
        if hasattr(self, '_hooks'):
            self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._cfg.hook))
        else:
            self._hooks = build_learner_hook_by_cfg(self._cfg.hook)

    def _setup_wrapper(self) -> None:
        """
        Overview:
            Use ``_time_wrapper`` to get ``train_time``.
        Note:
            ``data_time`` is wrapped in ``setup_dataloader``.
        """
        self._wrapper_timer = EasyTimer()
        self.train = self._time_wrapper(self.train, 'train_time')

    def _time_wrapper(self, fn: Callable, name: str) -> Callable:
        """
        Overview:
            Wrap a function and measure the time it used.
        Arguments:
            - fn (:obj:`Callable`): Function to be time_wrapped.
            - name (:obj:`str`): Name to be registered in ``_log_buffer``.
        Returns:
             - wrapper (:obj:`Callable`): The wrapper to acquire a function's time.
        """

        def wrapper(*args, **kwargs) -> Any:
            with self._wrapper_timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[name] = self._wrapper_timer.value
            return ret

        return wrapper

    def register_hook(self, hook: LearnerHook) -> None:
        """
        Overview:
            Add a new hook to learner.
        Arguments:
            - hook (:obj:`LearnerHook`): The hook to be added to learner.
        """
        add_learner_hook(self._hooks, hook)

    def train(self, data: dict) -> None:
        """
        Overview:
            Given training data, implement network update for one iteration and update related variables.
            Learner's API for serial entry. Also called in ``start`` for each iteration's training.
            "before_iter" and "after_iter" hooks are called once each.
        Arguments:
            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.
        Note:
            ``_policy`` must be set before calling this method.
            ``_policy.forward`` method contains: forward, backward, grad sync(if in distributed mode) and
            parameter update.
        """
        assert hasattr(self, '_policy'), "please set learner policy"
        self.call_hook('before_iter')
        # Pre-process data
        with self._timer:
            replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
            replay_unique_id = [d.get('replay_unique_id', None) for d in data]
            data = self._policy.data_preprocess(data)
        # Forward
        log_vars = self._policy.forward(data)
        # Update replay buffer's priority info
        priority = log_vars.pop('priority', None)
        self._priority_info = {
            'replay_buffer_idx': replay_buffer_idx,
            'replay_unique_id': replay_unique_id,
            'priority': priority
        }
        # Update log_buffer
        log_vars['data_preprocess_time'] = self._timer.value
        log_vars.update(self.collect_info)
        self._log_buffer.update(log_vars)

        self.call_hook('after_iter')
        self._last_iter.add(1)

    @auto_checkpoint
    def start(self) -> None:
        """
        Overview:
            Learner's API for parallel entry.
            For each iteration, learner will get data through ``_next_data`` and call ``train`` to train.
            Besides "before_iter" and "after_iter", "before_run" and "after_run" hooks are called once each.
        """
        self._end_flag = False
        self._finished_task = None
        # before run hook
        self.call_hook('before_run')

        max_iterations = self._cfg.max_iterations
        for _ in range(max_iterations):
            data = self._next_data()
            self.train(data)
            if self._end_flag:
                break

        self._finished_task = {'finish': True}
        # after run hook
        self.call_hook('after_run')

    def setup_dataloader(self) -> None:
        """
        Overview:
            Setup learner's dataloader.
        Note:
            Only in parallel version will we use ``get_data`` and ``_dataloader``(AsyncDataLoader);
            Instead in serial version, we can sample data from replay buffer directly.
            In parallel version, ``get_data`` is set by comm LearnerCommHelper, and should be callable.
            Users don't need to know the related details if not necessary.
        """
        cfg = self._cfg.dataloader
        self._dataloader = AsyncDataLoader(
            self.get_data,
            cfg.batch_size,
            self._device,
            cfg.chunk_size,
            collate_fn=lambda x: x,
            num_workers=cfg.num_workers
        )
        self._next_data = self._time_wrapper(self._next_data, 'data_time')

    def _next_data(self) -> Any:
        """
        Overview:
            Call ``_dataloader``'s ``__next__`` method to return next training data.
        Returns:
            - data (:obj:`Any`): Next training data from dataloader.
        Note:
            Only in parallel version will this method be called.
        """
        return next(self._dataloader)

    def close(self) -> None:
        """
        Overview:
            Close the related resources, e.g. dataloader, tensorboard logger, etc.
        """
        if self._end_flag:
            return
        self._end_flag = True
        if hasattr(self, '_dataloader'):
            del self._dataloader
        self._tb_logger.close()
        # if the learner is interrupted by force
        if self._finished_task is None:
            self.save_checkpoint()

    def call_hook(self, name: str) -> None:
        """
        Overview:
            Call the corresponding hook plugins according to position name.
        Arguments:
            - name (:obj:`str`): Hooks in which position to call, \
                should be in ['before_run', 'after_run', 'before_iter', 'after_iter'].
        """
        for hook in self._hooks[name]:
            hook(self)

    def info(self, s: str) -> None:
        """
        Overview:
            Log string info by ``self._logger.info``.
        Arguments:
            - s (:obj:`str`): The message to add into the logger.
        """
        self._logger.info(s)

    def debug(self, s: str) -> None:
        self._logger.debug(s)

    def save_checkpoint(self) -> None:
        """
        Overview:
            Directly call ``save_ckpt_after_run`` hook to save checkpoint.
        Note:
            Must guarantee that "save_ckpt_after_run" is registered in "after_run" hook.
            This method is called in:

                - ``auto_checkpoint``(``torch_utils/checkpoint_helper.py``), which is designed for \
                    saving checkpoint whenever an exception raises.
                - ``serial_pipeline``(``entry/serial_entry.py``). Used to save checkpoint after convergence or \
                    new highes reward during evaluation.
        """
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)

    @property
    def learn_info(self) -> dict:
        """
        Overview:
            Get current info dict, which will be sent to commander, e.g. replay buffer priority update,
            current iteration, hyper-parameter adjustment, whether task is finished, etc.
        Returns:
            - info (:obj:`dict`): Current learner info dict.
        """
        ret = {'learner_step': self._last_iter.val, 'priority_info': self._priority_info}
        if hasattr(self, '_finished_task') and self._finished_task is not None:
            ret['finished_task'] = self._finished_task
        return ret

    @property
    def last_iter(self) -> CountVar:
        return self._last_iter

    @property
    def train_iter(self) -> int:
        return self._last_iter.val

    @property
    def tick_time(self) -> TickTime:
        return self._tick_time

    @property
    def monitor(self) -> TickMonitor:
        return self._monitor

    @property
    def log_buffer(self) -> dict:  # LogDict
        return self._log_buffer

    @log_buffer.setter
    def log_buffer(self, _log_buffer: dict) -> None:
        self._log_buffer = _log_buffer

    @property
    def logger(self) -> 'TextLogger':  # noqa
        return self._logger

    @property
    def tb_logger(self) -> 'TensorBoradLogger':  # noqa
        return self._tb_logger

    @property
    def load_path(self) -> str:
        return self._load_path

    @load_path.setter
    def load_path(self, _load_path: str) -> None:
        self._load_path = _load_path

    @property
    def checkpoint_manager(self) -> Any:
        return self._checkpointer_manager

    @property
    def name(self) -> str:
        return self._name + str(id(self))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def policy(self) -> 'Policy':  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:  # noqa
        """
        Note:
            Monitor is set alongside with policy, because variables in monitor are determined by specific policy.
        """
        self._policy = _policy
        if self._rank == 0:
            self._monitor = get_simple_monitor_type(self._policy.monitor_vars())(TickTime(), expire=10)
        self.info(self._policy.info())

    @property
    def priority_info(self) -> dict:
        return self._priority_info

    @priority_info.setter
    def priority_info(self, _priority_info: dict) -> None:
        self._priority_info = _priority_info

    # ##################### only serial ###########################
    @property
    def collect_info(self) -> dict:
        return self._collect_info

    @collect_info.setter
    def collect_info(self, collect_info: dict) -> None:
        self._collect_info = {k: float(v) for k, v in collect_info.items()}


learner_mapping = {}


def register_learner(name: str, learner: type) -> None:
    """
    Overview:
        Add a new Learner class with its name to dict learner_mapping, any subclass derived from BaseLearner must
        use this function to register in nervex system before instantiate.
    Arguments:
        - name (:obj:`str`): Name of the new Learner.
        - learner (:obj:`type`): The new Learner class, should be subclass of ``BaseLearner``.
    """
    assert isinstance(name, str)
    assert issubclass(learner, BaseLearner)
    learner_mapping[name] = learner


def create_learner(cfg: EasyDict) -> BaseLearner:
    """
    Overview:
        Given the key(learner_name), create a new learner instance if in learner_mapping's values,
        or raise an KeyError. In other words, a derived learner must first register, then can call ``create_learner``
        to get the instance.
    Arguments:
        - cfg (:obj:`EasyDict`): Learner config. Necessary keys: [learner.import_module, learner.learner_type].
    Returns:
        - learner (:obj:`BaseLearner`): The created new learner, should be an instance of one of \
            learner_mapping's values.
    """
    import_module(cfg.import_names)
    learner_type = cfg.learner_type
    if learner_type not in learner_mapping.keys():
        raise KeyError("not support learner type: {}".format(learner_type))
    else:
        return learner_mapping[learner_type](cfg)
