"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning
"""
import os
from typing import Any, Union, Callable
from functools import partial
from easydict import EasyDict
import torch
from collections import namedtuple

from nervex.data import AsyncDataLoader, default_collate
from nervex.config import base_learner_default_config
from nervex.torch_utils import build_checkpoint_helper, CountVar, auto_checkpoint, build_log_buffer, to_device
from nervex.utils import build_logger, dist_init, EasyTimer, dist_finalize, pretty_print, read_config, \
    get_task_uid, import_module, broadcast
from nervex.utils import deep_merge_dicts
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook


class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info of one training iteration.
        Info include: cur_lr, time(data, train, forward, backward), loss(total,...)
        These info variables would first be recorded in ``log_buffer``, then in ``LearnerHook`` will vars in
        in this monitor be updated by``log_buffer``, then printed to ``TextLogger`` and ``TensorBoradLogger``.
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


def get_simple_monitor_type(properties: list = []):
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
        base class for model learning(SL/RL), which is able to multi-GPU learning
    Interface:
        __init__, run, close, call_hook, info, save_checkpoint, launch
    Property:
        last_iter, policy, log_buffer, record,
        load_path, save_path, checkpoint_manager, name, rank, tb_logger, use_distributed
    """

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            initialization method, load config setting and call ``_init`` for actual initialization,
            set the communication mode to `single_machine` or `flask_fs`.
            Use ``self._cfg`` setting to build common learner components, such as logger helper, checkpoint helper.
            launch learner runtime components, each train job means a launch operation,
            job related dataloader, policy and hook support.
        Arguments:
            - cfg (:obj:`EasyDict`): learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._cfg = deep_merge_dicts(base_learner_default_config, cfg)

        self._learner_worker_uid = get_task_uid()
        self._load_path = self._cfg.load_path
        self._save_path = self._cfg.save_path
        self._use_cuda = self._cfg.get('use_cuda', False)
        self._use_distributed = self._cfg.use_distributed
        if self._use_distributed:
            self._rank, self._world_size = dist_init()
            rand_id = torch.randint(0, 314, size=(1, ))
            broadcast(rand_id, 0)
            self._learner_uid = rand_id.item()
        else:
            self._rank, self._world_size = 0, 1
            self._learner_uid = self._learner_worker_uid
        self._timer = EasyTimer()
        self._device = 'cuda:{}'.format(self._rank % 8) if self._use_cuda else 'cpu'
        # monitor & logger
        # Only rank == 0 learner needs monitor and tb_logger, else only needs text_logger to display terminal output
        rank0 = True if self._rank == 0 else False
        path = os.path.join(self._cfg.save_path, 'learner')
        self._logger, self._tb_logger = build_logger(path, 'learner', rank0)
        self._log_buffer = build_log_buffer()
        # checkpoint helper
        self._checkpointer_manager = build_checkpoint_helper(self._cfg)
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        self._collect_info = {}
        self._priority_info = None
        self._last_iter = CountVar(init_val=0)
        self.info(pretty_print({
            "config": self._cfg,
        }, direct_print=False))

        # wrapper and hook
        self._setup_wrapper()
        self._setup_hook()

    def _setup_hook(self) -> None:
        """
        Overview:
            Setup hook for base_learner. Hook is the way to implement actual functions in base_learner.
            You can reference learner_hook.py
        """
        if hasattr(self, '_hooks'):
            self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._cfg.hook))
        else:
            self._hooks = build_learner_hook_by_cfg(self._cfg.hook)

    def _setup_wrapper(self) -> None:
        """
        Overview:
            Setup time_wrapper to get data_time and train_time
        """
        self._wrapper_timer = EasyTimer()
        self.train = self.time_wrapper(self.train, 'train_time')

    def time_wrapper(self, fn: Callable, name: str):
        """
        Overview:
            Wrap a function and measure the time it used
        Arguments:
            - fn (:obj:`Callable`): function to be time_wrapped
            - name (:obj:`str`): name to be registered in log_buffer
        """

        def wrapper(*args, **kwargs) -> Any:
            with self._wrapper_timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[name] = self._wrapper_timer.value
            return ret

        return wrapper

    @auto_checkpoint
    def train(self, data: dict) -> None:
        """
        Overview:
            Train the input data for 1 iteration, called in ``start`` which involves:
                - forward
                - backward
                - sync grad (if in distributed mode)
                - parameter update
        Arguments:
            - data (:obj:`dict`): data used for training
        """
        assert hasattr(self, '_policy'), "please set learner policy"
        self.call_hook('before_iter')
        replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
        replay_unique_id = [d.get('replay_unique_id', None) for d in data]
        with self._timer:
            data = self._policy.data_preprocess(data)
        log_vars = self._policy.forward(data)
        priority = log_vars.pop('priority', None)
        self._priority_info = {
            'replay_buffer_idx': replay_buffer_idx,
            'replay_unique_id': replay_unique_id,
            'priority': priority
        }
        log_vars['data_preprocess_time'] = self._timer.value
        log_vars.update(self.collect_info)
        self._log_buffer.update(log_vars)
        self.call_hook('after_iter')
        self._last_iter.add(1)

    def register_hook(self, hook: LearnerHook) -> None:
        """
        Overview:
            Add a new hook to learner.
        Arguments:
            - hook (:obj:`LearnerHook`): the hook to be added to learner
        """
        add_learner_hook(self._hooks, hook)

    def start(self) -> None:
        """
        Overview:
            Start to run the learner.
            For each iteration, learner will get training data and train.
            Learner will call hooks at four fixed positions(before_run, before_iter, after_iter, after_run).
        """
        self._finished_task = None
        # before run hook
        max_iterations = self._cfg.max_iterations
        self.call_hook('before_run')

        for _ in range(max_iterations):
            data = self._next_data()
            self.train(data)

        self._finished_task = {'finish': True}
        # after run hook
        self.call_hook('after_run')

    def setup_dataloader(self) -> None:
        """
        Overview:
            Setup learner's dataloader, get_data need to be a callable,
            and setup learner's collate_fn, which aggregate a listed data into a batch tensor.
        """
        # only in parallel version we use get_data and dataloader, instead in serial version,
        # main entry directly uses `train` method.
        # when parallel version, get_data is set by comm LearnerCommHelper
        # users don't need to know the related details if not necessary
        cfg = self._cfg.dataloader
        self._dataloader = AsyncDataLoader(
            self.get_data,
            cfg.batch_size,
            self._device,
            cfg.chunk_size,
            collate_fn=lambda x: x,
            num_workers=cfg.num_workers
        )
        self._next_data = self.time_wrapper(self._next_data, 'data_time')

    def _next_data(self) -> Any:
        return next(self._dataloader)

    def close(self) -> None:
        """
        Overview:
            Close the related resources, such as dist_finalize when use_distributed
        """
        if self._use_distributed:
            dist_finalize()
        if hasattr(self, '_dataloader'):
            del self._dataloader

    def call_hook(self, name: str) -> None:
        """
        Overview:
            Call the corresponding hook plugins according to name
        Arguments:
            - name (:obj:`str`): hooks in which position to call, \
                should be in ['before_run', 'after_run', 'before_iter', 'after_iter']
        """
        for hook in self._hooks[name]:
            hook(self)

    def info(self, s: str) -> None:
        """
        Overview:
            Log string info by ``self._logger.info``
        Arguments:
            - s (:obj:`str`): the message to add into the logger
        """
        self._logger.info(s)

    def save_checkpoint(self) -> None:
        """
        Overview:
            Automatically save checkpoints.
            Directly call ``save_ckpt_after_run`` hook instead of calling ``call_hook`` function.
        Note:
            This method is called by `auto_checkpoint` function in `checkpoint_helper.py`,
            designed for saving checkpoint whenever an exception raises.
        """
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)

    @property
    def learn_info(self) -> dict:
        """
        Overview:
            get learn info dict, which will be sent to commander for some operation, such as hyper-parameter adjustment
        Returns:
            info (:obj:`dict`): current info dict
        """
        ret = {'learner_step': self._last_iter.val, 'priority_info': self._priority_info}
        if hasattr(self, '_finished_task') and self._finished_task is not None:
            ret['finished_task'] = self._finished_task
        return ret

    @property
    def last_iter(self) -> CountVar:
        return self._last_iter

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
    def save_path(self) -> str:
        return self._save_path

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
        - name (:obj:`str`): name of the new Learner
        - learner (:obj:`type`): the new Learner class, should be subclass of BaseLearner
    """
    assert isinstance(name, str)
    assert issubclass(learner, BaseLearner)
    learner_mapping[name] = learner


def create_learner(cfg: EasyDict) -> BaseLearner:
    """
    Overview:
        Given the key(learner_type/name), create a new learner instance if in learner_mapping's values,
        or raise an KeyError. In other words, a derived learner must first register then call ``create_learner``
        to get the instance object.
    Arguments:
        - cfg (:obj:`EasyDict`): learner config, necessary keys: [learner.import_module, learner.learner_type]
    Returns:
        - learner (:obj:`BaseLearner`): the created new learner, should be an instance of one of\
            learner_mapping's values
    """
    import_module(cfg.learner.import_names)
    learner_type = cfg.learner.learner_type
    if learner_type not in learner_mapping.keys():
        raise KeyError("not support learner type: {}".format(learner_type))
    else:
        return learner_mapping[learner_type](cfg.learner)
