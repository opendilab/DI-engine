"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. checkpoint helper, used to help to save or load checkpoint by give args.
    2. CountVar, to help counting number.
"""
import logging
import os
import signal
import sys
import traceback

import torch

from sc2learner.utils.config_utils import save_config
from .data_helper import to_device

logger = logging.getLogger('default_logger')


def build_checkpoint_helper(save_path, rank=0):
    """
        Overview: use config to build checkpoint helper.
        Arguments:
            - cfg (:obj:`dict`): checkpoint_helper config
            - rank (:obj:`int`): creator process rank
        Returns:
            - (:obj`CheckpointHelper`): checkpoint_helper created by this function
    """
    return CheckpointHelper(save_path, rank)


class CheckpointHelper(object):
    """
        Overview: Concrete implementation of CheckpointHelper, to help to save or load checkpoint
        Interface: __init__, save_iterations, save, save_data, load
    """
    def __init__(self, save_dir='', rank=0):
        """
            Overview: initialization method, check if save_dir exists.
            Arguments:
                - save_dir (:obj:`str`): checkpoint save dir. if empty, saving is disabled
                - rank (:obs:`int`): creator process rank
        """
        if save_dir:
            self.save_path = os.path.join(save_dir, 'checkpoints')
            self.data_save_path = os.path.join(save_dir, 'data')
            if rank == 0:
                os.makedirs(self.save_path, exist_ok=True)
                os.makedirs(self.data_save_path, exist_ok=True)

    def _remove_prefix(self, state_dict, prefix='module.'):
        """
            Overview: remove prefix in state_dict
            Arguments:
                - state_dict (:obj:`dict`): model's state_dict
                - prefix (:obj:`str`): this prefix will be removed in keys
            Returns:
                - (:obj`dict`): new state_dict after removing prefix
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = ''.join(k.split(prefix))
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    def _add_prefix(self, state_dict, prefix='module.'):
        """
            Overview: add prefix in state_dict
            Arguments:
                - state_dict (:obj:`dict`): model's state_dict
                - prefix (:obj:`str`): this prefix will be added in keys
            Returns:
                - (:obj`dict`): new state_dict after adding prefix
        """
        return {prefix + k: v for k, v in state_dict.items()}

    def save_iterations(self, iterations, model, **kwargs):
        """
            Overview: save with iterations num
            Arguments:
                - iterations (:obj:`int`): iterations num
                - model (:obj:`str`): model to be saved
        """
        kwargs['last_iter'] = iterations
        return self.save('iterations_{}'.format(iterations), model, **kwargs)

    def save(
        self,
        name,
        model,
        optimizer=None,
        last_iter=None,
        last_epoch=None,
        dataset=None,
        actor_info=None,
        prefix_op=None,
        prefix=None
    ):
        """
            Overview: save checkpoint by given args
            Arguments:
                - name (:obj:`str`): checkpoint's name
                - model (:obj:`torch.nn.Module`): model to be saved
                - optimizer (:obj:`torch.optim.Optimizer`): optimizer obj
                - last_iter (:obj:`CountVar`): iter num, default zero
                - last_epoch (:obj:`CountVar`): epoch num, default zero
                - dataset (:obj:`torch.utils.data.Dataset`): dataset, should be replaydataset
                - actor_info (:obj:`torch.nn.Module`): attr of checkpoint, save actor info
                - prefix_op (:obj:`str`): should be ['remove', 'add'], process on state_dict
                - prefix (:obj:`str`): prefix to be processed on state_dict
        """
        checkpoint = {}
        state_dict = model.state_dict()
        if prefix_op is not None:  # remove or add prefix to state_dict.keys()
            prefix_func = {'remove': self._remove_prefix, 'add': self._add_prefix}
            if prefix_op not in prefix_func.keys():
                raise KeyError('invalid prefix_op:{}'.format(prefix_op))
            else:
                state_dict = prefix_func[prefix_op](state_dict, prefix)
        checkpoint['state_dict'] = state_dict

        if optimizer is not None:  # save optimizer
            assert (last_iter is not None or last_epoch is not None)
            checkpoint['last_iter'] = last_iter
            checkpoint['last_epoch'] = last_epoch
            checkpoint['optimizer'] = optimizer.state_dict()

        if dataset is not None:
            checkpoint['dataset'] = dataset.state_dict()
        if actor_info is not None:
            checkpoint['actor_info'] = actor_info.state_dict()
        path = os.path.join(self.save_path, name + '.pth.tar')
        torch.save(checkpoint, path)
        logger.info('save checkpoint in {}'.format(path))

    def save_data(self, name, data, device='cpu'):
        """
            Overview: save given tensor or dict
            Arguments:
                - name (:obj:`int`): file's name to be saved
                - data (:obj:`str`): data to be saved
                - device (:obj:`str`): save from gpu or cpu
        """
        assert (isinstance(data, torch.Tensor) or isinstance(data, dict))
        data = to_device(data, device)
        path = os.path.join(self.data_save_path, name + '_data.pt')
        torch.save(data, path)

    def save_config(self, config):
        save_config(config, os.path.join(self.data_save_path, "experiment_config.yaml"))

    def _load_matched_model_state_dict(self, model, ckpt_state_dict):
        """
            Overview: load matched model state_dict, and show mismatch keys between
                      model's state_dict and checkpoint's state_dict
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
        load_path,
        model,
        optimizer=None,
        last_iter=None,
        last_epoch=None,
        lr_schduler=None,
        dataset=None,
        actor_info=None,
        prefix_op=None,
        prefix=None,
        strict=True,
        logger_prefix='',
        state_dict_mask=[],
        need_torch_load=True,
    ):
        """
            Overview: load checkpoint by given path
            Arguments:
                - load_path (:obj:`str`): checkpoint's path
                - model (:obj:`torch.nn.Module`): model definition
                - optimizer (:obj:`Optimizer`): optimizer obj
                - last_iter (:obj:`CountVar`): iter num, default zero
                - last_epoch (:obj:`CountVar`): epoch num, default zero
                - lr_schduler (:obj:`Schduler`): lr_schduler obj
                - dataset (:obj:`Dataset`): dataset, should be replaydataset
                - actor_info (:obj:`torch.nn.Module`): attr of checkpoint, save actor info
                - prefix_op (:obj:`str`): should be ['remove', 'add'], process on state_dict
                - prefix (:obj:`str`): prefix to be processed on state_dict
                - strict (:obj:`bool`): args of model.load_state_dict
                - logger_prefix (:obj:`str`): prefix of logger
                - state_dict_mask (:obj:`list`) a list contains state_dict keys,
                    which shouldn't be loaded into model(after prefix op)
                - need_torch_load (:obj:`bool`): whether need torch.load operation
        """
        # Note: don't use assign operation('=') to update input argument value
        if isinstance(load_path, str):
            # don't assume this if input is not a path (like BytesIO for ceph)
            assert (os.path.exists(load_path))
        # Note: for reduce first GPU memory cost and compatible for cpu env
        if need_torch_load:
            checkpoint = torch.load(load_path, map_location='cpu')
        else:
            checkpoint = load_path
            load_path = 'object'
        state_dict = checkpoint['state_dict']
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
                )  # noqa
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
            dataset.load_state_dict(checkpoint['dataset'])
            logger.info(logger_prefix + 'load online data in {}'.format(load_path))

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(logger_prefix + 'load optimizer in {}'.format(load_path))

        if last_iter is not None:
            last_iter.update(checkpoint['last_iter'])
            logger.info(
                logger_prefix + 'load last_iter in {}, current last_iter is {}'.format(load_path, last_iter.val)
            )

        if last_epoch is not None:
            last_epoch.update(checkpoint['last_epoch'])
            logger.info(
                logger_prefix + 'load last_epoch in {}, current last_epoch is {}'.format(load_path, last_epoch.val)
            )  # noqa

        if actor_info is not None:
            actor_info.load_state_dict(checkpoint['actor_info'])
            logger.info(logger_prefix + 'load actor info in {}'.format(load_path))

        if lr_schduler is not None:
            assert (last_iter is not None)
            raise NotImplementedError


class CountVar(object):
    def __init__(self, init_val):
        self._val = init_val

    @property
    def val(self):
        return self._val

    def update(self, val):
        self._val = val

    def add(self, add_num):
        self._val += add_num


def auto_checkpoint(func):
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
        print('valid sig: ({})\tinvalid sig: ({})'.format(valid_sig, invalid_sig))

    def wrapper(*args, **kwargs):
        handle = args[0]
        assert (hasattr(handle, 'save_checkpoint'))

        def signal_handler(signal_num, frame):
            sig = signal.Signals(signal_num)
            print("SIGNAL: {}({})".format(sig.name, sig.value))
            handle.save_checkpoint()
            sys.exit(1)

        register_signal_handler(signal_handler)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handle.save_checkpoint()
            traceback.print_exc()

    return wrapper
