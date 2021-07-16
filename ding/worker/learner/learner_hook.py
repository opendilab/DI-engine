import numbers
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from easydict import EasyDict

import ding
from ding.utils import allreduce, read_file, save_file, get_rank


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

    .. note::

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
        self._load_path = ext_args['load_path']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Load checkpoint to learner. Checkpoint info includes policy state_dict and iter num.
        Arguments:
            - engine (:obj:`BaseLearner`): The BaseLearner to load checkpoint to.
        """
        path = self._load_path
        if path == '':  # not load
            return
        state_dict = read_file(path)
        if 'last_iter' in state_dict:
            last_iter = state_dict.pop('last_iter')
            engine.last_iter.update(last_iter)
        engine.policy.load_state_dict(state_dict)
        engine.info('{} load ckpt in {}'.format(engine.instance_name, path))


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
            Save checkpoint in corresponding path.
            Checkpoint info includes policy state_dict and iter num.
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner which needs to save checkpoint
        """
        if engine.rank == 0 and engine.last_iter.val % self._freq == 0:
            if engine.instance_name == 'learner':
                dirname = './{}/ckpt'.format(engine.exp_name)
            else:
                dirname = './{}/ckpt_{}'.format(engine.exp_name, engine.instance_name)
            if not os.path.exists(dirname):
                try:
                    os.mkdir(dirname)
                except FileExistsError:
                    pass
            ckpt_name = engine.ckpt_name if engine.ckpt_name else 'iteration_{}.pth.tar'.format(engine.last_iter.val)
            path = os.path.join(dirname, ckpt_name)
            state_dict = engine.policy.state_dict()
            state_dict.update({'last_iter': engine.last_iter.val})
            save_file(path, state_dict)
            engine.info('{} save ckpt in {}'.format(engine.instance_name, path))


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
        # Only show log for rank 0 learner
        if engine.rank != 0:
            for k in engine.log_buffer:
                engine.log_buffer[k].clear()
            return
        # For 'scalar' type variables: log_buffer -> tick_monitor -> monitor_time.step
        for k, v in engine.log_buffer['scalar'].items():
            setattr(engine.monitor, k, v)
        engine.monitor.time.step()

        iters = engine.last_iter.val
        if iters % self._freq == 0:
            engine.info("=== Training Iteration {} Result ===".format(iters))
            # For 'scalar' type variables: tick_monitor -> var_dict -> text_logger & tb_logger
            var_dict = {}
            log_vars = engine.policy.monitor_vars()
            attr = 'avg'
            for k in log_vars:
                k_attr = k + '_' + attr
                var_dict[k_attr] = getattr(engine.monitor, attr)[k]()
            engine.logger.info(engine.logger.get_tabulate_vars_hor(var_dict))
            for k, v in var_dict.items():
                engine.tb_logger.add_scalar('{}_iter/'.format(engine.instance_name) + k, v, iters)
                engine.tb_logger.add_scalar('{}_step/'.format(engine.instance_name) + k, v, engine._collector_envstep)
            # For 'histogram' type variables: log_buffer -> tb_var_dict -> tb_logger
            tb_var_dict = {}
            for k in engine.log_buffer['histogram']:
                new_k = '{}/'.format(engine.instance_name) + k
                tb_var_dict[new_k] = engine.log_buffer['histogram'][k]
            for k, v in tb_var_dict.items():
                engine.tb_logger.add_histogram(k, v, iters)
        for k in engine.log_buffer:
            engine.log_buffer[k].clear()


class LogReduceHook(LearnerHook):
    """
    Overview:
        Hook to reduce the distributed(multi-gpu) logs
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
            reduce the logs from distributed(multi-gpu) learners
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
                if ding.enable_linklink:
                    allreduce(new_data)
                else:
                    new_data = new_data.to(get_rank())
                    allreduce(new_data)
                    new_data = new_data.cpu()
            elif isinstance(data, numbers.Integral) or isinstance(data, numbers.Real):
                new_data = torch.scalar_tensor(data).reshape([1])
                if ding.enable_linklink:
                    allreduce(new_data)
                else:
                    new_data = new_data.to(get_rank())
                    allreduce(new_data)
                    new_data = new_data.cpu()
                new_data = new_data.item()
            else:
                raise TypeError("invalid type in reduce: {}".format(type(data)))
            return new_data

        engine.log_buffer = aggregate(engine.log_buffer)


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


simplified_hook_mapping = {
    'log_show_after_iter': lambda freq: hook_mapping['log_show']
    ('log_show', 20, position='after_iter', ext_args=EasyDict({'freq': freq})),
    'load_ckpt_before_run': lambda path: hook_mapping['load_ckpt']
    ('load_ckpt', 20, position='before_run', ext_args=EasyDict({'load_path': path})),
    'save_ckpt_after_iter': lambda freq: hook_mapping['save_ckpt']
    ('save_ckpt_after_iter', 20, position='after_iter', ext_args=EasyDict({'freq': freq})),
    'save_ckpt_after_run': lambda _: hook_mapping['save_ckpt']('save_ckpt_after_run', 20, position='after_run'),
    'log_reduce_after_iter': lambda _: hook_mapping['log_reduce']('log_reduce_after_iter', 10, position='after_iter'),
}


def find_char(s: str, flag: str, num: int, reverse: bool = False) -> int:
    assert num > 0, num
    count = 0
    iterable_obj = reversed(range(len(s))) if reverse else range(len(s))
    for i in iterable_obj:
        if s[i] == flag:
            count += 1
            if count == num:
                return i
    return -1


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
    for key, value in cfg.items():
        if key in simplified_hook_mapping and not isinstance(value, dict):
            pos = key[find_char(key, '_', 2, reverse=True) + 1:]
            hook = simplified_hook_mapping[key](value)
            priority = hook.priority
        else:
            priority = value.get('priority', 100)
            pos = value.position
            ext_args = value.get('ext_args', {})
            hook = hook_mapping[value.type](value.name, priority, position=pos, ext_args=ext_args)
        idx = 0
        for i in reversed(range(len(hooks[pos]))):
            if priority >= hooks[pos][i].priority:
                idx = i + 1
                break
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


def show_hooks(hooks: Dict[str, List[Hook]]) -> None:
    for k in hooks.keys():
        print('{}: {}'.format(k, [x.__class__.__name__ for x in hooks[k]]))
