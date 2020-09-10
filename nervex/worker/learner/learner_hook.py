from abc import ABC, abstractmethod
import numbers
import os
import torch
from typing import Any
from easydict import EasyDict
from nervex.utils import allreduce


class Hook(ABC):
    def __init__(self, name: str, priority: float, **kwargs) -> None:
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
        raise NotImplementedError


class LearnerHook(Hook):
    positions = ['before_run', 'after_run', 'before_iter', 'after_iter']

    def __init__(self, *args, position: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert position in self.positions
        self._position = position


class LrSchdulerHook(LearnerHook):
    def __init__(self, *args, **kwargs) -> None:
        ext_args = kwargs.pop('ext_args')
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        if engine.last_iter.val % self._freq == 0:
            engine.lr_scheduler.step()
        # for the normal case that all the parameters have the same lr
        engine.log_buffer['cur_lr'] = engine.lr_scheduler.get_lr()[0]


class LoadCkptHook(LearnerHook):
    def __init__(self, *args, **kwargs) -> None:
        ext_args = kwargs.pop('ext_args')
        super().__init__(*args, **kwargs)

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        path = engine.load_path
        if path == '':  # not load
            return
        engine.checkpoint_manager.load(
            path,
            model=engine.optimizer.agent.model,
            optimizer=engine.optimizer,
            last_iter=engine.last_iter,
            logger_prefix='({})'.format(engine.name),
        )
        engine.info('{} load ckpt in {}'.format(engine.name, path))


class SaveCkptHook(LearnerHook):
    def __init__(self, *args, **kwargs) -> None:
        ext_args = kwargs.pop('ext_args')
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        if engine.rank == 0 and engine.last_iter.val % self._freq == 0:
            dirname = os.path.join(engine.save_path, 'ckpt')
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            path = os.path.join(dirname, 'iteration_{}.pth.tar'.format(engine.last_iter.val))
            engine.checkpoint_manager.save(
                path,
                model=engine.optimizer.agent.model,
                optimizer=engine.optimizer,
                last_iter=engine.last_iter,
            )
            engine.info('{} save ckpt in {}'.format(engine.name, path))


class LogShowHook(LearnerHook):
    def __init__(self, *args, **kwargs) -> None:
        ext_args = kwargs.pop('ext_args')
        super().__init__(*args, **kwargs)
        if ext_args == {}:
            self._freq = 1
        else:
            self._freq = ext_args.freq

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        if engine.rank != 0:  # only show log at rank 0
            engine.log_buffer = {}  # reset log buffer
            return
        engine.record.update_var(engine.log_buffer)
        engine.log_buffer = {}
        iters = engine.last_iter.val
        if iters % self._freq == 0:
            engine.info("=== Training Iteration {} Result ===".format(iters))
            engine.info(engine.record.get_vars_text())
            tb_keys = engine.tb_logger.scalar_var_names
            engine.tb_logger.add_val_list(
                engine.record.get_vars_tb_format(tb_keys, iters, var_type='scalar'), viz_type='scalar'
            )


class LogReduceHook(LearnerHook):
    def __init__(self, *args, **kwargs) -> None:
        ext_args = kwargs.pop('ext_args')
        super().__init__(*args, **kwargs)

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        assert engine.use_distributed

        def aggregate(data):
            r"""
            Overview:
                aggregate the information from all ranks(usually use sync allreduce)
            Arguments:
                - data (:obj:`dict`): data needs to be reduced. Could be dict, torch.Tensor,
                numbers.Integral or numbers.Real.
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


def build_learner_hook_by_cfg(cfg: EasyDict):
    """
    Note: lower value means higher priority
    """
    hook_mapping = {
        'lr_scheduler': LrSchdulerHook,
        'load_ckpt': LoadCkptHook,
        'save_ckpt': SaveCkptHook,
        'log_show': LogShowHook,
        'log_reduce': LogReduceHook,
    }
    hooks = {k: [] for k in LearnerHook.positions}
    for item in cfg.values():
        priority = item.get('priority', 0)
        pos = item.position
        idx = 0
        for i in reversed(range(len(hooks[pos]))):
            if priority >= hooks[pos][i].priority:
                idx = i + 1
                break
        ext_args = item.get('ext_args', {})
        hook = hook_mapping[item.type](item.name, priority, position=pos, ext_args=ext_args)
        hooks[item.position].insert(idx, hook)
    return hooks
