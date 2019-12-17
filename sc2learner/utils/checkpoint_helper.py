import os
import torch
import logging
from .data_helper import to_device


logger = logging.getLogger('default_logger')


def build_checkpoint_helper(cfg, rank=0):
    if rank == 0:
        return CheckpointHelper(cfg.common.save_path)
    else:
        return None


class CheckpointHelper(object):
    def __init__(self, save_dir):
        self.save_path = os.path.join(save_dir, 'checkpoints')
        self.data_save_path = os.path.join(save_dir, 'data')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.data_save_path):
            os.mkdir(self.data_save_path)

    def _remove_prefix(self, state_dict, prefix='module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = ''.join(k.split(prefix))
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    def _add_prefix(self, state_dict, prefix='module.'):
        return {prefix+k: v for k, v in state_dict.items()}

    def save_iterations(self, iterations, model, **kwargs):
        kwargs['last_iter'] = iterations
        return self.save('iterations_{}'.format(iterations), model, **kwargs)

    def save(self, name, model,
             optimizer=None, last_iter=None,
             prefix_op=None, prefix=None):
        checkpoint = {}
        state_dict = model.state_dict()
        if prefix_op is not None:
            prefix_func = {'remove': self._remove_prefix,
                           'add': self._add_prefix}
            if prefix_op not in prefix_func.keys():
                raise KeyError('invalid prefix_op:{}'.format(prefix_op))
            else:
                state_dict = prefix_func[prefix_op](state_dict, prefix)
        checkpoint['state_dict'] = state_dict

        if optimizer is not None:
            assert(last_iter is not None)
            checkpoint['last_iter'] = last_iter
            checkpoint['optimizer'] = optimizer.state_dict()
        path = os.path.join(self.save_path, name+'.pth.tar')
        torch.save(checkpoint, path)
        logger.info('save checkpoint in {}'.format(path))

    def save_data(self, name, data, device='cpu'):
        assert(isinstance(data, torch.Tensor) or isinstance(data, dict))
        data = to_device(data, device)
        path = os.path.join(self.data_save_path, name+'_data.pt')
        torch.save(data, path)

    def _print_mismatch_keys(self, model_state_dict, ckpt_state_dict):
        model_keys = set(model_state_dict.keys())
        ckpt_keys = set(ckpt_state_dict.keys())
        miss_keys = model_keys - ckpt_keys
        redundant_keys = ckpt_keys - model_keys
        for k in miss_keys:
            logger.info('miss_keys: {}'.format(k))
        for k in redundant_keys:
            logger.info('redundant_keys: {}'.format(k))

    def load(self, load_path, model,
             optimizer=None, last_iter=None, lr_schduler=None,
             prefix_op=None, prefix=None, strict=False, logger_prefix=''):
        # Note: don't use assign operation('=') to updare input argument value
        assert(os.path.exists(load_path))
        checkpoint = torch.load(load_path)
        state_dict = checkpoint['state_dict']
        if prefix_op is not None:
            prefix_func = {'remove': self._remove_prefix,
                           'add': self._add_prefix}
            if prefix_op not in prefix_func.keys():
                raise KeyError('invalid prefix_op:{}'.format(prefix_op))
            else:
                state_dict = prefix_func[prefix_op](state_dict, prefix)
        model.load_state_dict(state_dict, strict=strict)
        logger.info(logger_prefix+'load model state_dict in {}'.format(load_path))
        self._print_mismatch_keys(model.state_dict(), state_dict)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(logger_prefix+'load optimizer in {}'.format(load_path))

        if last_iter is not None:
            last_iter.update(checkpoint['last_iter'])
            logger.info(logger_prefix+'load last_iter in {}, current last_iter is {}'.format(load_path, last_iter.val))

        if lr_schduler is not None:
            assert(last_iter is not None)
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
