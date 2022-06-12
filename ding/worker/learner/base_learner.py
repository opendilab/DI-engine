from typing import Any, Union, Callable, List, Dict, Optional, Tuple
from ditk import logging
from collections import namedtuple
from functools import partial
from easydict import EasyDict

import copy

from ding.torch_utils import CountVar, auto_checkpoint, build_log_buffer
from ding.utils import build_logger, EasyTimer, import_module, LEARNER_REGISTRY, get_rank, get_world_size
from ding.utils.autolog import LoggedValue, LoggedModel, TickTime
from ding.utils.data import AsyncDataLoader
from .learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook


@LEARNER_REGISTRY.register('base')
class BaseLearner(object):
    r"""
    Overview:
        Base class for policy learning.
    Interface:
        train, call_hook, register_hook, save_checkpoint, start, setup_dataloader, close
    Property:
        learn_info, priority_info, last_iter, train_iter, rank, world_size, policy
        monitor, log_buffer, logger, tb_logger, ckpt_name, exp_name, instance_name
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        train_iterations=int(1e9),
        dataloader=dict(num_workers=0, ),
        log_policy=True,
        # --- Hooks ---
        hook=dict(
            load_ckpt_before_run='',
            log_show_after_iter=100,
            save_ckpt_after_iter=10000,
            save_ckpt_after_run=True,
        ),
    )

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(
            self,
            cfg: EasyDict,
            policy: namedtuple = None,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
            dist_info: Tuple[int, int] = None,
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'learner',
    ) -> None:
        """
        Overview:
            Initialization method, build common learner components according to cfg, such as hook, wrapper and so on.
        Arguments:
            - cfg (:obj:`EasyDict`): Learner config, you can refer cls.config for details.
            - policy (:obj:`namedtuple`): A collection of policy function of learn mode. And policy can also be \
                initialized when runtime.
            - tb_logger (:obj:`SummaryWriter`): Tensorboard summary writer.
            - dist_info (:obj:`Tuple[int, int]`): Multi-GPU distributed training information.
            - exp_name (:obj:`str`): Experiment name, which is used to indicate output directory.
            - instance_name (:obj:`str`): Instance name, which should be unique among different learners.
        Notes:
            If you want to debug in sync CUDA mode, please add the following code at the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._cfg = cfg
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._ckpt_name = None
        self._timer = EasyTimer()

        # These 2 attributes are only used in parallel mode.
        self._end_flag = False
        self._learner_done = False
        if dist_info is None:
            self._rank = get_rank()
            self._world_size = get_world_size()
        else:
            # Learner rank. Used to discriminate which GPU it uses.
            self._rank, self._world_size = dist_info
        if self._world_size > 1:
            self._cfg.hook.log_reduce_after_iter = True

        # Logger (Monitor will be initialized in policy setter)
        # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
        if self._rank == 0:
            if tb_logger is not None:
                self._logger, _ = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
                )
                self._tb_logger = tb_logger
            else:
                self._logger, self._tb_logger = build_logger(
                    './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name
                )
        else:
            self._logger, _ = build_logger(
                './{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False
            )
            self._tb_logger = None
        self._log_buffer = {
            'scalar': build_log_buffer(),
            'scalars': build_log_buffer(),
            'histogram': build_log_buffer(),
        }

        # Setup policy
        if policy is not None:
            self.policy = policy

        # Learner hooks. Used to do specific things at specific time point. Will be set in ``_setup_hook``
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        # Last iteration. Used to record current iter.
        self._last_iter = CountVar(init_val=0)

        # Setup time wrapper and hook.
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
        self.train = self._time_wrapper(self.train, 'scalar', 'train_time')

    def _time_wrapper(self, fn: Callable, var_type: str, var_name: str) -> Callable:
        """
        Overview:
            Wrap a function and record the time it used in ``_log_buffer``.
        Arguments:
            - fn (:obj:`Callable`): Function to be time_wrapped.
            - var_type (:obj:`str`): Variable type, e.g. ['scalar', 'scalars', 'histogram'].
            - var_name (:obj:`str`): Variable name, e.g. ['cur_lr', 'total_loss'].
        Returns:
             - wrapper (:obj:`Callable`): The wrapper to acquire a function's time.
        """

        def wrapper(*args, **kwargs) -> Any:
            with self._wrapper_timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[var_type][var_name] = self._wrapper_timer.value
            return ret

        return wrapper

    def register_hook(self, hook: LearnerHook) -> None:
        """
        Overview:
            Add a new learner hook.
        Arguments:
            - hook (:obj:`LearnerHook`): The hook to be addedr.
        """
        add_learner_hook(self._hooks, hook)

    def train(self, data: dict, envstep: int = -1, policy_kwargs: Optional[dict] = None) -> None:
        """
        Overview:
            Given training data, implement network update for one iteration and update related variables.
            Learner's API for serial entry.
            Also called in ``start`` for each iteration's training.
        Arguments:
            - data (:obj:`dict`): Training data which is retrieved from repaly buffer.

        .. note::

            ``_policy`` must be set before calling this method.

            ``_policy.forward`` method contains: forward, backward, grad sync(if in multi-gpu mode) and
            parameter update.

            ``before_iter`` and ``after_iter`` hooks are called at the beginning and ending.
        """
        assert hasattr(self, '_policy'), "please set learner policy"
        self.call_hook('before_iter')

        if policy_kwargs is None:
            policy_kwargs = {}

        # Forward
        log_vars = self._policy.forward(data, **policy_kwargs)

        # Update replay buffer's priority info
        if isinstance(log_vars, dict):
            priority = log_vars.pop('priority', None)
        elif isinstance(log_vars, list):
            priority = log_vars[-1].pop('priority', None)
        else:
            raise TypeError("not support type for log_vars: {}".format(type(log_vars)))
        if priority is not None:
            replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data]
            replay_unique_id = [d.get('replay_unique_id', None) for d in data]
            self.priority_info = {
                'priority': priority,
                'replay_buffer_idx': replay_buffer_idx,
                'replay_unique_id': replay_unique_id,
            }
        # Discriminate vars in scalar, scalars and histogram type
        # Regard a var as scalar type by default. For scalars and histogram type, must annotate by prefix "[xxx]"
        self._collector_envstep = envstep
        if isinstance(log_vars, dict):
            log_vars = [log_vars]
        for elem in log_vars:
            scalars_vars, histogram_vars = {}, {}
            for k in list(elem.keys()):
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    scalars_vars[new_k] = elem.pop(k)
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    histogram_vars[new_k] = elem.pop(k)
            # Update log_buffer
            self._log_buffer['scalar'].update(elem)
            self._log_buffer['scalars'].update(scalars_vars)
            self._log_buffer['histogram'].update(histogram_vars)

            self.call_hook('after_iter')
            self._last_iter.add(1)

        return log_vars

    @auto_checkpoint
    def start(self) -> None:
        """
        Overview:
            [Only Used In Parallel Mode] Learner's API for parallel entry.
            For each iteration, learner will get data through ``_next_data`` and call ``train`` to train.

        .. note::

            ``before_run`` and ``after_run`` hooks are called at the beginning and ending.
        """
        self._end_flag = False
        self._learner_done = False
        # before run hook
        self.call_hook('before_run')

        for i in range(self._cfg.train_iterations):
            data = self._next_data()
            if self._end_flag:
                break
            self.train(data)

        self._learner_done = True
        # after run hook
        self.call_hook('after_run')

    def setup_dataloader(self) -> None:
        """
        Overview:
            [Only Used In Parallel Mode] Setup learner's dataloader.

        .. note::

            Only in parallel mode will we use attributes ``get_data`` and ``_dataloader`` to get data from file system;
            Instead, in serial version, we can fetch data from memory directly.

            In parallel mode, ``get_data`` is set by ``LearnerCommHelper``, and should be callable.
            Users don't need to know the related details if not necessary.
        """
        cfg = self._cfg.dataloader
        batch_size = self._policy.get_attribute('batch_size')
        device = self._policy.get_attribute('device')
        chunk_size = cfg.chunk_size if 'chunk_size' in cfg else batch_size
        self._dataloader = AsyncDataLoader(
            self.get_data, batch_size, device, chunk_size, collate_fn=lambda x: x, num_workers=cfg.num_workers
        )
        self._next_data = self._time_wrapper(self._next_data, 'scalar', 'data_time')

    def _next_data(self) -> Any:
        """
        Overview:
            [Only Used In Parallel Mode] Call ``_dataloader``'s ``__next__`` method to return next training data.
        Returns:
            - data (:obj:`Any`): Next training data from dataloader.
        """
        return next(self._dataloader)

    def close(self) -> None:
        """
        Overview:
            [Only Used In Parallel Mode] Close the related resources, e.g. dataloader, tensorboard logger, etc.
        """
        if self._end_flag:
            return
        self._end_flag = True
        if hasattr(self, '_dataloader'):
            self._dataloader.close()
        if self._tb_logger:
            self._tb_logger.flush()
            self._tb_logger.close()

    def __del__(self) -> None:
        self.close()

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
        self._logger.info('[RANK{}]: {}'.format(self._rank, s))

    def debug(self, s: str) -> None:
        self._logger.debug('[RANK{}]: {}'.format(self._rank, s))

    def save_checkpoint(self, ckpt_name: str = None) -> None:
        """
        Overview:
            Directly call ``save_ckpt_after_run`` hook to save checkpoint.
        Note:
            Must guarantee that "save_ckpt_after_run" is registered in "after_run" hook.
            This method is called in:

                - ``auto_checkpoint`` (``torch_utils/checkpoint_helper.py``), which is designed for \
                    saving checkpoint whenever an exception raises.
                - ``serial_pipeline`` (``entry/serial_entry.py``). Used to save checkpoint when reaching \
                    new highest evaluation reward.
        """
        if ckpt_name is not None:
            self.ckpt_name = ckpt_name
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)
        self.ckpt_name = None

    @property
    def learn_info(self) -> dict:
        """
        Overview:
            Get current info dict, which will be sent to commander, e.g. replay buffer priority update,
            current iteration, hyper-parameter adjustment, whether task is finished, etc.
        Returns:
            - info (:obj:`dict`): Current learner info dict.
        """
        ret = {
            'learner_step': self._last_iter.val,
            'priority_info': self.priority_info,
            'learner_done': self._learner_done
        }
        return ret

    @property
    def last_iter(self) -> CountVar:
        return self._last_iter

    @property
    def train_iter(self) -> int:
        return self._last_iter.val

    @property
    def monitor(self) -> 'TickMonitor':  # noqa
        return self._monitor

    @property
    def log_buffer(self) -> dict:  # LogDict
        return self._log_buffer

    @log_buffer.setter
    def log_buffer(self, _log_buffer: Dict[str, Dict[str, Any]]) -> None:
        self._log_buffer = _log_buffer

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def tb_logger(self) -> 'TensorBoradLogger':  # noqa
        return self._tb_logger

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def instance_name(self) -> str:
        return self._instance_name

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def policy(self) -> 'Policy':  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:  # noqa
        """
        Note:
            Policy variable monitor is set alongside with policy, because variables are determined by specific policy.
        """
        self._policy = _policy
        if self._rank == 0:
            self._monitor = get_simple_monitor_type(self._policy.monitor_vars())(TickTime(), expire=10)
        if self._cfg.log_policy:
            self.info(self._policy.info())

    @property
    def priority_info(self) -> dict:
        if not hasattr(self, '_priority_info'):
            self._priority_info = {}
        return self._priority_info

    @priority_info.setter
    def priority_info(self, _priority_info: dict) -> None:
        self._priority_info = _priority_info

    @property
    def ckpt_name(self) -> str:
        return self._ckpt_name

    @ckpt_name.setter
    def ckpt_name(self, _ckpt_name: str) -> None:
        self._ckpt_name = _ckpt_name


def create_learner(cfg: EasyDict, **kwargs) -> BaseLearner:
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
    import_module(cfg.get('import_names', []))
    return LEARNER_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)


class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info during training.
        Info includes: cur_lr, time(data, train, forward, backward), loss(total,...)
        These info variables are firstly recorded in ``log_buffer``, then in ``LearnerHook`` will vars in
        in this monitor be updated by``log_buffer``, finally printed to text logger and tensorboard logger.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    data_time = LoggedValue(float)
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
            return sum(_list) / len(_list) if len(_list) != 0 else 0

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
            'data_time', 'train_time', 'sample_count', 'total_collect_step', 'total_step', 'total_sample',
            'total_episode', 'total_duration'
        ] + properties
        for p_name in properties:
            attrs[p_name] = LoggedValue(float)
        return type('SimpleTickMonitor', (TickMonitor, ), attrs)
