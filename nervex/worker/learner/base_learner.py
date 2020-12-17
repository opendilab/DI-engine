"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Union, Callable
from functools import partial
from easydict import EasyDict
import torch
from collections import namedtuple

from nervex.data import AsyncDataLoader, default_collate
from nervex.torch_utils import build_checkpoint_helper, CountVar, auto_checkpoint, build_log_buffer, to_device
from nervex.utils import build_logger, dist_init, EasyTimer, dist_finalize, pretty_print, read_config, \
    get_task_uid, import_module, broadcast
from nervex.utils import deep_merge_dicts
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from .comm import LearnerCommHelper
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook

default_config = read_config(os.path.join(os.path.dirname(__file__), "base_learner_default_config.yaml"))


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
    cur_lr = LoggedValue(float)
    data_time = LoggedValue(float)
    train_time = LoggedValue(float)
    forward_time = LoggedValue(float)
    backward_time = LoggedValue(float)
    total_loss = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        self.register_attribute_value('avg', 'cur_lr', partial(__avg_func, prop_name='cur_lr'))
        self.register_attribute_value('avg', 'data_time', partial(__avg_func, prop_name='data_time'))
        self.register_attribute_value('avg', 'train_time', partial(__avg_func, prop_name='train_time'))
        self.register_attribute_value('avg', 'forward_time', partial(__avg_func, prop_name='forward_time'))
        self.register_attribute_value('avg', 'backward_time', partial(__avg_func, prop_name='backward_time'))
        self.register_attribute_value('avg', 'total_loss', partial(__avg_func, prop_name='total_loss'))


class BaseLearner(ABC):
    r"""
    Overview:
        base class for model learning(SL/RL), which is able to multi-GPU learning
    Interface:
        __init__, register_stats, run, close, call_hook, info, save_checkpoint, launch
    Property:
        last_iter, optimizer, lr_scheduler, computation_graph, agent, log_buffer, record,
        load_path, save_path, checkpoint_manager, name, rank, tb_logger, use_distributed
    """

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            initialization method, load config setting and call ``_init`` for actual initialization,
            set the communication mode to `single_machine` or `flask_fs`.
        Arguments:
            - cfg (:obj:`EasyDict`): learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._cfg = deep_merge_dicts(default_config, cfg)
        self._init()
        if self._cfg.learner.communication.type == 'single_machine':
            self._logger.info("Single machine learner has launched")
        else:
            comm_cfg = self._cfg.learner.communication
            LearnerCommHelper.enable_comm_helper(self, comm_cfg)
            self._logger.info("Distributed learner has launched")

    def _init(self) -> None:
        """
        Overview:
            Use ``self._cfg`` setting to build common learner components, such as dataset, runtime agent,
            optimizer, lr_scheduler, logger helper, checkpoint helper.
        """
        self._learner_worker_uid = get_task_uid()
        self._load_path = self._cfg.common.load_path
        self._save_path = self._cfg.common.save_path
        self._use_cuda = self._cfg.learner.use_cuda
        self._use_distributed = self._cfg.learner.use_distributed
        if self._use_distributed:
            self._rank, self._world_size = dist_init()
            rand_id = torch.randint(0, 314, size=(1, ))
            broadcast(rand_id, 0)
            self._learner_uid = rand_id.item()
        else:
            self._rank, self._world_size = 0, 1
            self._learner_uid = self._learner_worker_uid
        if self._use_cuda:
            self._device = 'cuda: {}'.format(self._rank % 8)
        else:
            self._device = 'cpu'
        self._default_max_iterations = self._cfg.learner.max_iterations
        self._timer = EasyTimer()
        # monitor & logger
        # Only rank == 0 learner needs monitor and tb_logger, else only needs text_logger to display terminal output
        rank0 = True if self._rank == 0 else False
        self._monitor = TickMonitor(TickTime(), expire=10) if rank0 else None
        path = os.path.join(self._cfg.common.save_path, 'learner')
        self._logger, self._tb_logger = build_logger(path, 'learner', rank0)
        self._log_buffer = build_log_buffer()
        # checkpoint helper
        self._checkpointer_manager = build_checkpoint_helper(self._cfg)
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        self._collate_fn = default_collate

    def launch(self) -> None:
        """
        Overview:
            launch learner runtime components, each train job means a launch operation,
            job related dataloader, agent, computation_graph, optimizer and hook support.
        """
        self._setup_dataloader()
        self._setup_agent()
        self._setup_optimizer()
        self._setup_computation_graph()

        self._last_iter = CountVar(init_val=0)
        if self._rank == 0:
            self.register_stats()
        self.info(
            pretty_print(
                {
                    "config": self._cfg,
                    "agent": repr(self._agent),
                    "computation_graph": repr(self._computation_graph)
                },
                direct_print=False
            )
        )

        self._setup_wrapper()
        self._setup_hook()

    def _setup_hook(self) -> None:
        """
        Overview:
            Setup hook for base_learner. Hook is the way to implement actual functions in base_learner.
            You can reference learner_hook.py
        """
        if hasattr(self, '_hooks'):
            self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._cfg.learner.hook))
        else:
            self._hooks = build_learner_hook_by_cfg(self._cfg.learner.hook)

    def _setup_wrapper(self) -> None:
        """
        Overview:
            Setup time_wrapper to get data_time and train_time
        """
        self._wrapper_timer = EasyTimer()
        self._get_iter_data = self.time_wrapper(self._get_iter_data, 'data_time')
        self._train = self.time_wrapper(self._train, 'train_time')

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

    def _setup_dataloader(self) -> None:
        """
        Overview:
            Setup learner's dataloader, data_source need to be a generator,
            and setup learner's collate_fn, which aggregate a listed data into a batch tensor.
        """
        cfg = self._cfg.learner.data
        # when single machine version, get_data is set by SingleMachineRunner
        # when distributed version, get_data is set by comm LearnerCommHelper
        # users don't need to know the related details if not necessary
        self._dataloader = AsyncDataLoader(
            self.get_data, cfg.batch_size, self._device, cfg.chunk_size, self._collate_fn, cfg.num_workers
        )

    def _get_iter_data(self) -> Any:
        return next(self._dataloader)

    @abstractmethod
    def _setup_agent(self) -> None:
        """
        Overview:
            Setup learner's runtime agent, agent is the subclass instance of `BaseAgent`.
            There may be more than one agent.
        Note:
            `agent` is the wrapped `model`, it can be wrapped with different plugins to satisfy
            different runtime usages (e.g. actor and learner would use the model differently)
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_computation_graph(self) -> None:
        """
        Overview:
            Setup computation_graph, which uses procssed data and agent to get an optimization
            computation graph.
        """
        raise NotImplementedError

    def _setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer and lr_scheduler
        """
        self._optimizer = torch.optim.Adam(
            self._agent.model.parameters(),
            lr=self._cfg.learner.learning_rate,
            weight_decay=self._cfg.learner.weight_decay
        )
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[], gamma=1)

    def _train(self, data: Any) -> None:
        """
        Overview:
            Train the input data for 1 iteration, called in ``run`` which involves:

                - forward
                - backward
                - sync grad (if in distributed mode)
                - parameter update
        Arguments:
            - data (:obj:`Any`): data used for training
        """
        with self._timer:
            log_vars = self._computation_graph.forward(data, self._agent)
            loss = log_vars['total_loss']
        self._log_buffer['forward_time'] = self._timer.value

        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
            if self._use_distributed:
                self._agent.model.sync_gradients()
            self._optimizer.step()
        self._log_buffer['backward_time'] = self._timer.value
        self._log_buffer.update(log_vars)

    def register_stats(self) -> None:
        """
        Overview:
            Register some basic attributes to tb_logger (e.g.: cur_lr, data_time, train_time).
            Pass tb_logger to computation_graph for more registration.
        """
        self._tb_logger.register_var('cur_lr_avg')
        self._tb_logger.register_var('data_time_avg')
        self._tb_logger.register_var('train_time_avg')
        self._tb_logger.register_var('forward_time_avg')
        self._tb_logger.register_var('backward_time_avg')
        self._computation_graph.register_stats(self._tb_logger)

    def register_hook(self, hook: LearnerHook) -> None:
        """
        Overview:
            Add a new hook to learner.
        Arguments:
            - hook (:obj:`LearnerHook`): the hook to be added to learner
        """
        add_learner_hook(self._hooks, hook)

    @auto_checkpoint
    def run(self, max_iterations: Union[int, None] = None) -> None:
        """
        Overview:
            Run the learner.
            For each iteration, learner will get training data and train.
            Learner will call hooks at four fixed positions(before_run, before_iter, after_iter, after_run).
        Arguments:
            - max_iterations (:obj:`int`): the max run iteration, if None then set to default_max_iterations
        """
        if max_iterations is None:
            max_iterations = self._default_max_iterations
        # before run hook
        self.call_hook('before_run')

        for _ in range(max_iterations):
            data = self._get_iter_data()
            # before iter hook
            self.call_hook('before_iter')
            self._train(data)
            # after iter hook
            self.call_hook('after_iter')
            self._last_iter.add(1)

        # after run hook
        self.call_hook('after_run')

    def close(self) -> None:
        """
        Overview:
            Close the related resources, such as dist_finalize when use_distributed
        """
        if self._use_distributed:
            dist_finalize()

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
    def last_iter(self) -> CountVar:
        return self._last_iter

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._lr_scheduler

    @property
    def computation_graph(self) -> Any:
        return self._computation_graph

    @property
    def agent(self) -> 'BaseAgent':  # noqa
        return self._agent

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
    def use_distributed(self) -> bool:
        return self._use_distributed


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
        return learner_mapping[learner_type](cfg)
