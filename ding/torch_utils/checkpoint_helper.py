from ditk import logging
import signal
import sys
import traceback
from typing import Callable
import torch
import torch.utils.data  # torch1.1.0 compatibility
from ding.utils import read_file, save_file

logger = logging.getLogger('default_logger')


def build_checkpoint_helper(cfg):
    r"""
    Overview:
        Use config to build checkpoint helper.
    Arguments:
        - cfg (:obj:`dict`): ckpt_helper config
    Returns:
        - (:obj:`CheckpointHelper`): checkpoint_helper created by this function
    """
    return CheckpointHelper()


class CheckpointHelper:
    r"""
    Overview:
        Help to save or load checkpoint by give args.
    Interface:
        save, load
    """

    def __init__(self):
        pass

    def _remove_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        r"""
        Overview:
            Remove prefix in state_dict
        Arguments:
            - state_dict (:obj:`dict`): model's state_dict
            - prefix (:obj:`str`): this prefix will be removed in keys
        Returns:
            - new_state_dict (:obj:`dict`): new state_dict after removing prefix
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = ''.join(k.split(prefix))
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    def _add_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        r"""
        Overview:
            Add prefix in state_dict
        Arguments:
            - state_dict (:obj:`dict`): model's state_dict
            - prefix (:obj:`str`): this prefix will be added in keys
        Returns:
            - (:obj:`dict`): new state_dict after adding prefix
        """
        return {prefix + k: v for k, v in state_dict.items()}

    def save(
            self,
            path: str,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer = None,
            last_iter: 'CountVar' = None,  # noqa
            last_epoch: 'CountVar' = None,  # noqa
            last_frame: 'CountVar' = None,  # noqa
            dataset: torch.utils.data.Dataset = None,
            collector_info: torch.nn.Module = None,
            prefix_op: str = None,
            prefix: str = None,
    ) -> None:
        r"""
        Overview:
            Save checkpoint by given args
        Arguments:
            - path (:obj:`str`): the path of saving checkpoint
            - model (:obj:`torch.nn.Module`): model to be saved
            - optimizer (:obj:`torch.optim.Optimizer`): optimizer obj
            - last_iter (:obj:`CountVar`): iter num, default None
            - last_epoch (:obj:`CountVar`): epoch num, default None
            - last_frame (:obj:`CountVar`): frame num, default None
            - dataset (:obj:`torch.utils.data.Dataset`): dataset, should be replaydataset
            - collector_info (:obj:`torch.nn.Module`): attr of checkpoint, save collector info
            - prefix_op (:obj:`str`): should be ['remove', 'add'], process on state_dict
            - prefix (:obj:`str`): prefix to be processed on state_dict
        """
        checkpoint = {}
        model = model.state_dict()
        if prefix_op is not None:  # remove or add prefix to model.keys()
            prefix_func = {'remove': self._remove_prefix, 'add': self._add_prefix}
            if prefix_op not in prefix_func.keys():
                raise KeyError('invalid prefix_op:{}'.format(prefix_op))
            else:
                model = prefix_func[prefix_op](model, prefix)
        checkpoint['model'] = model

        if optimizer is not None:  # save optimizer
            assert (last_iter is not None or last_epoch is not None)
            checkpoint['last_iter'] = last_iter.val
            if last_epoch is not None:
                checkpoint['last_epoch'] = last_epoch.val
            if last_frame is not None:
                checkpoint['last_frame'] = last_frame.val
            checkpoint['optimizer'] = optimizer.state_dict()

        if dataset is not None:
            checkpoint['dataset'] = dataset.state_dict()
        if collector_info is not None:
            checkpoint['collector_info'] = collector_info.state_dict()
        save_file(path, checkpoint)
        logger.info('save checkpoint in {}'.format(path))

    def _load_matched_model_state_dict(self, model: torch.nn.Module, ckpt_state_dict: dict) -> None:
        r"""
        Overview:
            Load matched model state_dict, and show mismatch keys between model's state_dict and checkpoint's state_dict
        Arguments:
            - model (:obj:`torch.nn.Module`): model
            - ckpt_state_dict (:obj:`dict`): checkpoint's state_dict
        """
        assert isinstance(model, torch.nn.Module)
        diff = {'miss_keys': [], 'redundant_keys': [], 'mismatch_shape_keys': []}
        model_state_dict = model.state_dict()
        model_keys = set(model_state_dict.keys())
        ckpt_keys = set(ckpt_state_dict.keys())
        diff['miss_keys'] = model_keys - ckpt_keys
        diff['redundant_keys'] = ckpt_keys - model_keys

        intersection_keys = model_keys.intersection(ckpt_keys)
        valid_keys = []
        for k in intersection_keys:
            if model_state_dict[k].shape == ckpt_state_dict[k].shape:
                valid_keys.append(k)
            else:
                diff['mismatch_shape_keys'].append(
                    '{}\tmodel_shape: {}\tckpt_shape: {}'.format(
                        k, model_state_dict[k].shape, ckpt_state_dict[k].shape
                    )
                )
        valid_ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if k in valid_keys}
        model.load_state_dict(valid_ckpt_state_dict, strict=False)

        for n, keys in diff.items():
            for k in keys:
                logger.info('{}: {}'.format(n, k))

    def load(
        self,
        load_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        last_iter: 'CountVar' = None,  # noqa
        last_epoch: 'CountVar' = None,  # noqa
        last_frame: 'CountVar' = None,  # noqa
        lr_schduler: 'Scheduler' = None,  # noqa
        dataset: torch.utils.data.Dataset = None,
        collector_info: torch.nn.Module = None,
        prefix_op: str = None,
        prefix: str = None,
        strict: bool = True,
        logger_prefix: str = '',
        state_dict_mask: list = [],
    ):
        r"""
        Overview:
            Load checkpoint by given path
        Arguments:
            - load_path (:obj:`str`): checkpoint's path
            - model (:obj:`torch.nn.Module`): model definition
            - optimizer (:obj:`torch.optim.Optimizer`): optimizer obj
            - last_iter (:obj:`CountVar`): iter num, default None
            - last_epoch (:obj:`CountVar`): epoch num, default None
            - last_frame (:obj:`CountVar`): frame num, default None
            - lr_schduler (:obj:`Schduler`): lr_schduler obj
            - dataset (:obj:`torch.utils.data.Dataset`): dataset, should be replaydataset
            - collector_info (:obj:`torch.nn.Module`): attr of checkpoint, save collector info
            - prefix_op (:obj:`str`): should be ['remove', 'add'], process on state_dict
            - prefix (:obj:`str`): prefix to be processed on state_dict
            - strict (:obj:`bool`): args of model.load_state_dict
            - logger_prefix (:obj:`str`): prefix of logger
            - state_dict_mask (:obj:`list`): A list containing state_dict keys, \
                which shouldn't be loaded into model(after prefix op)

        .. note::

            The checkpoint loaded from load_path is a dict, whose format is like '{'state_dict': OrderedDict(), ...}'
        """
        # TODO save config
        # Note: for reduce first GPU memory cost and compatible for cpu env
        checkpoint = read_file(load_path)
        state_dict = checkpoint['model']
        if prefix_op is not None:
            prefix_func = {'remove': self._remove_prefix, 'add': self._add_prefix}
            if prefix_op not in prefix_func.keys():
                raise KeyError('invalid prefix_op:{}'.format(prefix_op))
            else:
                state_dict = prefix_func[prefix_op](state_dict, prefix)
        if len(state_dict_mask) > 0:
            if strict:
                logger.info(
                    logger_prefix +
                    '[Warning] non-empty state_dict_mask expects strict=False, but finds strict=True in input argument'
                )
                strict = False
            for m in state_dict_mask:
                state_dict_keys = list(state_dict.keys())
                for k in state_dict_keys:
                    if k.startswith(m):
                        state_dict.pop(k)  # ignore return value
        if strict:
            model.load_state_dict(state_dict, strict=True)
        else:
            self._load_matched_model_state_dict(model, state_dict)
        logger.info(logger_prefix + 'load model state_dict in {}'.format(load_path))

        if dataset is not None:
            if 'dataset' in checkpoint.keys():
                dataset.load_state_dict(checkpoint['dataset'])
                logger.info(logger_prefix + 'load online data in {}'.format(load_path))
            else:
                logger.info(logger_prefix + "dataset not in checkpoint, ignore load procedure")

        if optimizer is not None:
            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info(logger_prefix + 'load optimizer in {}'.format(load_path))
            else:
                logger.info(logger_prefix + "optimizer not in checkpoint, ignore load procedure")

        if last_iter is not None:
            if 'last_iter' in checkpoint.keys():
                last_iter.update(checkpoint['last_iter'])
                logger.info(
                    logger_prefix + 'load last_iter in {}, current last_iter is {}'.format(load_path, last_iter.val)
                )
            else:
                logger.info(logger_prefix + "last_iter not in checkpoint, ignore load procedure")

        if collector_info is not None:
            collector_info.load_state_dict(checkpoint['collector_info'])
            logger.info(logger_prefix + 'load collector info in {}'.format(load_path))

        if lr_schduler is not None:
            assert (last_iter is not None)
            raise NotImplementedError


class CountVar(object):
    r"""
    Overview:
        Number counter
    Interface:
        val, update, add
    """

    def __init__(self, init_val: int) -> None:
        self._val = init_val

    @property
    def val(self) -> int:
        return self._val

    def update(self, val: int) -> None:
        r"""
        Overview:
            Update the var counter
        Arguments:
            - val (:obj:`int`): the update value of the counter
        """
        self._val = val

    def add(self, add_num: int):
        r"""
        Overview:
            Add the number to counter
        Arguments:
            - add_num (:obj:`int`): the number added to the counter
        """
        self._val += add_num


def auto_checkpoint(func: Callable) -> Callable:
    r"""
    Overview:
        Create a wrapper to wrap function, and the wrapper will call the save_checkpoint method
        whenever an exception happens.
    Arguments:
        - func(:obj:`Callable`): the function to be wrapped
    Returns:
        - wrapper (:obj:`Callable`): the wrapped function
    """
    dead_signals = ['SIGILL', 'SIGINT', 'SIGKILL', 'SIGQUIT', 'SIGSEGV', 'SIGSTOP', 'SIGTERM', 'SIGBUS']
    all_signals = dead_signals + ['SIGUSR1']

    def register_signal_handler(handler):
        valid_sig = []
        invalid_sig = []
        for sig in all_signals:
            try:
                sig = getattr(signal, sig)
                signal.signal(sig, handler)
                valid_sig.append(sig)
            except Exception:
                invalid_sig.append(sig)
        logger.info('valid sig: ({})\ninvalid sig: ({})'.format(valid_sig, invalid_sig))

    def wrapper(*args, **kwargs):
        handle = args[0]
        assert (hasattr(handle, 'save_checkpoint'))

        def signal_handler(signal_num, frame):
            sig = signal.Signals(signal_num)
            logger.info("SIGNAL: {}({})".format(sig.name, sig.value))
            handle.save_checkpoint('ckpt_interrupt.pth.tar')
            sys.exit(1)

        register_signal_handler(signal_handler)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handle.save_checkpoint('ckpt_exception.pth.tar')
            traceback.print_exc()

    return wrapper
