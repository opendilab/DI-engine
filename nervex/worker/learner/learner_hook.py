import numbers
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from easydict import EasyDict

from nervex.utils import allreduce


class Hook(ABC):
    """
    Overview:
        Abstract class for hooks.
    Interfaces:
        __init__, __call__
    Property:
        name, priority
    """

    def __init__(self, name: str, priority: float, **kwargs) -> None:
        """
        Overview:
            Init method for hooks. Set name and priority.
        Arguments:
            - name (:obj:`str`): The name of hook
            - priority (:obj:`float`): The priority used in ``call_hook``'s calling sequence. \
                Lower value means higher priority.
        """
        self._name = name
        assert priority >= 0, "invalid priority value: {}".format(priority)
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> float:
        return self._priority

    @abstractmethod
    def __call__(self, engine: Any) -> Any:
        """
        Overview:
            Should be overwritten by subclass.
        Arguments:
            - engine (:obj:`Any`): For LearnerHook, it should be ``BaseLearner`` or its subclass.
        """
        raise NotImplementedError


class LearnerHook(Hook):
    """
    Overview:
        Abstract class for hooks used in Learner.
    Interfaces:
        __init__
    Property:
        name, priority, position
    Note:
        Subclass should implement ``self.__call__``.
    """
    positions = ['before_run', 'after_run', 'before_iter', 'after_iter']

    def __init__(self, *args, position: str, **kwargs) -> None:
        """
        Overview:
            Init LearnerHook.
        Arguments:
            - position (:obj:`str`): The position to call hook in learner. \
                Must be in ['before_run', 'after_run', 'before_iter', 'after_iter'].
        """
        super().__init__(*args, **kwargs)
        assert position in self.positions
        self._position = position

    @property
    def position(self) -> str:
        return self._position


class LoadCkptHook(LearnerHook):
    """
    Overview:
        Hook to load checkpoint
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict = EasyDict(), **kwargs) -> None:
        """
        Overview:
            Init LoadCkptHook.
        Arguments:
            - ext_args (:obj:`EasyDict`): Extended arguments. Use ``ext_args.freq`` to set ``load_ckpt_freq``.
        """
        super().__init__(*args, **kwargs)

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Load checkpoint to learner.
        Arguments:
            - engine (:obj:`BaseLearner`): The BaseLearner to load checkpoint to.
        """
        path = engine.load_path
        if path == '':  # not load
            return
        policy_handle = engine.policy.state_dict_handle()
        optimizer = policy_handle.get('optimizer', None)
        engine.checkpoint_manager.load(
            path,
            model=policy_handle['model'],
            optimizer=optimizer,
            last_iter=engine.last_iter,
            logger_prefix='({})'.format(engine.name),
        )
        engine.info('{} load ckpt in {}'.format(engine.name, path))


class SaveCkptHook(LearnerHook):
    """
    Overview:
        Hook to save checkpoint
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict = EasyDict(), **kwargs) -> None:
        """
        Overview:
            init SaveCkptHook
        Arguments:
            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set save_ckpt_freq
        """
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Save check point in corresponding path, using ``engine.checkpoint_manager``
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner which needs to save checkpoint
        """
        if engine.rank == 0 and engine.last_iter.val % self._freq == 0:
            dirname = './ckpt{}'.format(engine.name)
            if not os.path.exists(dirname):
                try:
                    os.mkdir(dirname)
                except FileExistsError:
                    pass
            path = os.path.join(dirname, 'iteration_{}.pth.tar'.format(engine.last_iter.val))
            policy_handle = engine.policy.state_dict_handle()
            optimizer = policy_handle.get('optimizer', None)
            engine.checkpoint_manager.save(
                path,
                model=policy_handle['model'],
                optimizer=optimizer,
                last_iter=engine.last_iter,
            )
            engine.info('{} save ckpt in {}'.format(engine.name, path))


class LogShowHook(LearnerHook):
    """
    Overview:
        Hook to show log
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict = EasyDict(), **kwargs) -> None:
        """
        Overview:
            init LogShowHook
        Arguments:
            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set freq
        """
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Show log, update record and tb_logger if rank is 0 and at interval iterations,
            clear the log buffer for all learners regardless of rank
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        if engine.rank != 0:  # only show log at rank 0
            engine.log_buffer.clear()  # reset log buffer
            return
        # log_buffer -> tick_monitor -> monitor_time.step
        for k, v in engine.log_buffer.items():
            setattr(engine.monitor, k, v)
        engine.monitor.time.step()

        iters = engine.last_iter.val
        if iters % self._freq == 0:
            engine.info("=== Training Iteration {} Result ===".format(iters))
            # tick_monitor -> var_dict
            var_dict = {}
            for k in engine.log_buffer:
                for attr in engine.monitor.get_property_attribute(k):
                    k_attr = k + '_' + attr
                    var_dict[k_attr] = getattr(engine.monitor, attr)[k]()
            engine.logger.print_vars(var_dict)
            engine.tb_logger.print_vars(var_dict, iters, 'scalar')
        engine.log_buffer.clear()


class LogReduceHook(LearnerHook):
    """
    Overview:
        Hook to reduce the distributed logs
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: EasyDict = EasyDict(), **kwargs) -> None:
        """
        Overview:
            init LogReduceHook
        Arguments:
            - ext_args (:obj:`EasyDict`): extended_args, use ext_args.freq to set log_reduce_freq
        """
        super().__init__(*args, **kwargs)

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            reduce the logs from distributed learners
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """

        def aggregate(data):
            r"""
            Overview:
                aggregate the information from all ranks(usually use sync allreduce)
            Arguments:
                - data (:obj:`dict`): Data that needs to be reduced. \
                    Could be dict, torch.Tensor, numbers.Integral or numbers.Real.
            Returns:
                - new_data (:obj:`dict`): data after reduce
            """
            if isinstance(data, dict):
                new_data = {k: aggregate(v) for k, v in data.items()}
            elif isinstance(data, list) or isinstance(data, tuple):
                new_data = [aggregate(t) for t in data]
            elif isinstance(data, torch.Tensor):
                new_data = data.clone().detach()
                allreduce(new_data)  # get data from other processes
            elif isinstance(data, numbers.Integral) or isinstance(data, numbers.Real):
                new_data = torch.scalar_tensor(data).reshape([1])
                allreduce(new_data)
                new_data = new_data.item()
            else:
                raise TypeError("invalid type in reduce: {}".format(type(data)))
            return new_data

        engine.log_buffer = aggregate(engine.buffer)


hook_mapping = {
    'load_ckpt': LoadCkptHook,
    'save_ckpt': SaveCkptHook,
    'log_show': LogShowHook,
    'log_reduce': LogReduceHook,
}


def register_learner_hook(name: str, hook_type: type) -> None:
    """
    Overview:
        Add a new LearnerHook class to hook_mapping, so you can build one instance with `build_learner_hook_by_cfg`.
        You can reference
        <https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/blob/master/nervex/worker/learner/tests/test_base_learner.py#L81>
        or see Example below
    Arguments:
        - name (:obj:`str`): name of the register hook
        - hook_type (:obj:`type`): the register hook_type you implemented that realize LearnerHook
    Examples:
        >>> class HookToRegister(LearnerHook):
        >>>     def __init__(*args, **kargs):
        >>>         ...
        >>>         ...
        >>>     def __call__(*args, **kargs):
        >>>         ...
        >>>         ...
        >>> ...
        >>> register_learner_hook('name_of_hook', HookToRegister)
        >>> ...
        >>> hooks = build_learner_hook_by_cfg(cfg)
    """
    assert issubclass(hook_type, LearnerHook)
    hook_mapping[name] = hook_type


def build_learner_hook_by_cfg(cfg: EasyDict) -> Dict[str, List[Hook]]:
    """
    Overview:
        Build the learner hooks in hook_mapping by config.
        This function is often used to initialize ``hooks`` according to cfg,
        while add_learner_hook() is often used to add an existing LearnerHook to `hooks`.
    Arguments:
        - cfg (:obj:`EasyDict`): Config dict. Should be like {'hook': xxx}.
    Returns:
        - hooks (:obj:`Dict[str, List[Hook]`): Keys should be in ['before_run', 'after_run', 'before_iter', \
            'after_iter'], each value should be a list containing all hooks in this position.
    Note:
        Lower value means higher priority.
    """
    hooks = {k: [] for k in LearnerHook.positions}
    for item in cfg.values():
        priority = item.get('priority', 100)
        pos = item.position
        idx = 0
        for i in reversed(range(len(hooks[pos]))):
            if priority >= hooks[pos][i].priority:
                idx = i + 1
                break
        ext_args = item.get('ext_args', {})
        hook = hook_mapping[item.type](item.name, priority, position=pos, ext_args=ext_args)
        hooks[pos].insert(idx, hook)
    return hooks


def add_learner_hook(hooks: Dict[str, List[Hook]], hook: LearnerHook) -> None:
    """
    Overview:
        Add a learner hook(:obj:`LearnerHook`) to hooks(:obj:`Dict[str, List[Hook]`)
    Arguments:
        - hooks (:obj:`Dict[str, List[Hook]`): You can refer to ``build_learner_hook_by_cfg``'s return ``hooks``.
        - hook (:obj:`LearnerHook`): The LearnerHook which will be added to ``hooks``.
    """
    position = hook.position
    priority = hook.priority
    idx = 0
    for i in reversed(range(len(hooks[position]))):
        if priority >= hooks[position][i].priority:
            idx = i + 1
            break
    assert isinstance(hook, LearnerHook)
    hooks[position].insert(idx, hook)


def merge_hooks(hooks1: Dict[str, List[Hook]], hooks2: Dict[str, List[Hook]]) -> Dict[str, List[Hook]]:
    """
    Overview:
        Merge two hooks dict, which have the same keys, and each value is sorted by hook priority with stable method.
    Arguments:
        - hooks1 (:obj:`Dict[str, List[Hook]`): hooks1 to be merged.
        - hooks2 (:obj:`Dict[str, List[Hook]`): hooks2 to be merged.
    Returns:
        - new_hooks (:obj:`Dict[str, List[Hook]`): New merged hooks dict.
    Note:
        This merge function uses stable sort method without disturbing the same priority hook.
    """
    assert set(hooks1.keys()) == set(hooks2.keys())
    new_hooks = {}
    for k in hooks1.keys():
        new_hooks[k] = sorted(hooks1[k] + hooks2[k], key=lambda x: x.priority)
    return new_hooks
