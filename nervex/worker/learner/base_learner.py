"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning
"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Union
import yaml
import os.path as osp
from easydict import EasyDict
import torch
from nervex.torch_utils import build_checkpoint_helper, CountVar, auto_checkpoint, build_log_buffer, to_device
from nervex.utils import build_logger, dist_init, EasyTimer, dist_finalize, pretty_print, merge_dicts, read_config
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, LearnerHook

default_config = read_config(osp.join(osp.dirname(__file__), "base_learner_default_config.yaml"))


class BaseLearner(ABC):
    r"""
    Overview:
        base class for model learning(SL/RL), which uses linklink for multi-GPU learning
    Interface:
        __init__, _setup_hook, _setup_wrapper, time_wrapper, _setup_data_source, _setup_computation_graph
        _setup_optimizer, _get_data, _train, register_stats, run, close, call_hook, info, save_checkpoint
    """

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            initialization method, using config setting to build model, dataset, optimizer, lr_scheduler
            and other helper. It can also load and save checkpoint.
        Arguments:
            - cfg (:obj:`dict`): learner config, you can view `learner_cfg <../../../configuration/index.html>`_\
            for reference
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of `__init__`.

            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._cfg = merge_dicts(default_config, cfg)
        self._load_path = self._cfg.common.load_path
        self._save_path = self._cfg.common.save_path
        self._use_cuda = self._cfg.learner.use_cuda
        self._use_distributed = self._cfg.learner.use_distributed
        if self._use_distributed:
            self._rank, self._world_size = dist_init()
        else:
            self._rank, self._world_size = 0, 1
        self._default_max_iterations = self._cfg.learner.max_iterations
        self._last_iter = CountVar(init_val=0)
        self._timer = EasyTimer()

        self._setup_data_source()
        self._setup_computation_graph()
        self._setup_optimizer()

        # logger
        self._logger, self._tb_logger, self._record = build_logger(self._cfg, rank=self._rank)
        self._log_buffer = build_log_buffer()
        # checkpoint helper
        self._checkpointer_manager = build_checkpoint_helper(self._cfg, rank=self._rank)
        self.register_stats()
        self.info(
            pretty_print({
                "config": self._cfg,
                "computation_graph": repr(self._computation_graph)
            }, direct_print=False)
        )

        self._setup_wrapper()
        self._setup_hook()

    def _setup_hook(self) -> None:
        """
        Overview:
            Setup hook for base_learner. Hook is the way to implement actually functions in base_learner.
            You can reference learner_hook.py
        """
        self._hooks = build_learner_hook_by_cfg(self._cfg.learner.hook)

    def _setup_wrapper(self) -> None:
        """
        Overview:
            Setup time_wrapper to get the data_time and train_time
        """
        self._get_data = self.time_wrapper(self._get_data, 'data_time')
        self._train = self.time_wrapper(self._train, 'train_time')

    def time_wrapper(self, fn, name):
        """
        Overview:
            The time_wrapper used to get the time a function used
        """
        def wrapper(*args, **kwargs):
            with self._timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[name] = self._timer.value
            return ret

        return wrapper

    @abstractmethod
    def _setup_data_source(self) -> None:
        """
        Overview:
            Setup learner's data_source, data_source need to be iterable
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_computation_graph(self) -> None:
        """
        Overview:
            Setup computation_graph, used as loss calculater, part of the optimizer
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer
        """
        raise NotImplementedError

    def _get_data(self) -> Any:
        """
        Overview:
            get data from data_source
        """
        data = next(self._data_source)
        if self._use_cuda:
            data = to_device(data, 'cuda')
        return data

    def _train(self, data: Any) -> dict:
        """
        Overview:
            train the input data for 1 iteration
        """
        # Note
        # processes: forward -> backward -> sync grad(only dist) -> update param
        with self._timer:
            log_vars = self._computation_graph.forward(data)
            loss = log_vars['total_loss']
        self._log_buffer['forward_time'] = self._timer.value

        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
            if self._use_distributed:
                self._computation_graph.sync_gradients()
            self._optimizer.step()
        self._log_buffer['backward_time'] = self._timer.value
        self._log_buffer.update(log_vars)

    def register_stats(self) -> None:
        """
        Overview:
            register cur_lr, data_time, train_time to record, and register record to optimizer
        """
        self._record.register_var('cur_lr')
        self._record.register_var('data_time')
        self._record.register_var('train_time')
        self._record.register_var('forward_time')
        self._record.register_var('backward_time')

        self._tb_logger.register_var('cur_lr')
        self._tb_logger.register_var('data_time')
        self._tb_logger.register_var('train_time')
        self._tb_logger.register_var('forward_time')
        self._tb_logger.register_var('backward_time')

        self._computation_graph.register_stats(self._record, self._tb_logger)

    def register_hook(self, hook: LearnerHook):
        add_learner_hook(self._hooks, hook)

    @auto_checkpoint
    def run(self, max_iterations: Union[int, None] = None) -> None:
        """
        Overview:
            Run the learner
        Arguments:
            - max_iterations (:obj:`int` or :obj:`None`): the max run iteration,
                if None then set to default_max_iterations
        """
        if max_iterations is None:
            max_iterations = self._default_max_iterations
        # before run hook
        self.call_hook('before_run')

        for _ in range(max_iterations):
            data = self._get_data()
            # before iter hook
            self.call_hook('before_iter')
            log_vars = self._train(data)
            # after iter hook
            self.call_hook('after_iter')
            self._last_iter.add(1)

        # after run hook
        self.call_hook('after_run')

    def close(self) -> None:
        """
        Overview:
            Close linklink if use_distributed
        """
        if self._use_distributed:
            dist_finalize()

    def call_hook(self, name: str) -> None:
        """
        Overview:
            Call the corresponding hook plugins according to name
        Arguments:
             - name (:obj:`str`): which hooks to call,
                should be in ['before_run', 'after_run', 'before_iter', 'after_iter']
        """
        for hook in self._hooks[name]:
            hook(self)

    def info(self, s: str) -> None:
        """
        Overview:
            Return logger.info
        """
        self._logger.info(s)

    def save_checkpoint(self) -> None:
        """
        Overview:
            Automatically save checkpoints
        Note:
            this method is designed for auto_checkpoint
        """
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)

    @property
    def last_iter(self) -> CountVar:
        return self._last_iter

    @abstractproperty
    def optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractproperty
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        raise NotImplementedError

    @abstractproperty
    def computation_graph(self) -> Any:
        raise NotImplementedError

    @property
    def log_buffer(self) -> dict:  # LogDict
        return self._log_buffer

    @log_buffer.setter
    def log_buffer(self, _log_buffer: dict) -> None:
        self._log_buffer = _log_buffer

    @property
    def record(self) -> 'VariableRecord':  # noqa
        return self._record

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
    def tb_logger(self) -> 'TensorBoardLogger':  # noqa
        return self._tb_logger

    @property
    def use_distributed(self) -> bool:
        return self._use_distributed
