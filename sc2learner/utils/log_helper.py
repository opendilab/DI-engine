'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. log helper, used to help to save logger on terminal, tensorboard or save file.
    2. CountVar, to help counting number.
'''
import logging
import numbers
import numpy as np
import cv2
import os
from tensorboardX import SummaryWriter


def build_logger(cfg, name=None, rank=0):
    '''
        Overview: use config to build checkpoint helper. Only rank == 0 can build.
        Arguments:
            - name (:obj:`str`): logger file name
            - rank (:obj:`int`): only rank == 0 can build
        Returns:
            - (:obj`TextLogger`): save terminal output
            - (:obj`TensorBoardLogger`): save output to tensorboard
            - (:obj`VariableRecord`): record variable for further process
    '''
    # Note: Only support rank0 logger
    if rank == 0:
        path = cfg.common.save_path
        logger = TextLogger(path, name=name)
        tb_logger = TensorBoardLogger(path, name=name)
        var_record_type = cfg.logger.get("var_record_type", None)
        if var_record_type is None:
            variable_record = VariableRecord(cfg.logger.print_freq)
        elif var_record_type == 'alphastar':
            variable_record = AlphastarVarRecord(cfg.logger.print_freq)
        else:
            raise NotImplementedError("not support var_record_type: {}".format(var_record_type))
        return logger, tb_logger, variable_record
    else:
        return None, None, None


def get_default_logger(name=None):
    if name is None:
        name = 'default_logger'
    return logging.getLogger(name)


class TextLogger(object):
    '''
        Overview: save terminal output to file
        Interface: __init__, info
    '''
    def __init__(self, path, name=None):
        '''
            Overview: initialization method, create logger.
            Arguments:
                - path (:obj:`str`): logger's save dir
                - name (:obj:`str`): logger's name
        '''
        if name is None:
            name = 'default_logger'
        self.logger = self._create_logger(name, os.path.join(path, name+'.txt'))

    def _create_logger(self, name, path, level=logging.INFO):
        '''
            Overview: create logger using logging
            Arguments:
                - name (:obj:`str`): logger's name
                - path (:obj:`str`): logger's save dir
            Returns:
                - (:obj`logger`): new logger
        '''
        logger = logging.getLogger(name)
        if not logger.handlers:
            formatter = logging.Formatter(
                '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(path, 'a')
            fh.setFormatter(formatter)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger

    def info(self, s):
        '''
            Overview: add message to logger
            Arguments:
                - s (:obj:`str`): message to add to logger
        '''
        self.logger.info(s)


class TensorBoardLogger(object):
    '''
        Overview: save message to tensorboard
        Interface: __init__, add_scalar, add_text, add_scalars, add_histogram, add_figure,
                   add_image, add_scalar_list, register_var, scalar_var_names
    '''
    def __init__(self, path, name=None):
        '''
            Overview: initialization method, create logger and set var names.
            Arguments:
                - path (:obj:`str`): logger save dir
                - name (:obj:`str`): logger name
        '''
        if name is None:
            name = 'default_tb_logger'
        self.logger = SummaryWriter(os.path.join(path, name))  # get summary writer
        self._var_names = {
            'scalar': [],
            'text': [],
            'scalars': [],
            'histogram': [],
            'figure': [],
            'image': [],
        }

    def add_scalar(self, name, *args, **kwargs):
        '''
            Overview: add message to scalar
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['scalar']
        '''
        assert(name in self._var_names['scalar'])
        self.logger.add_scalar(name, *args, **kwargs)

    def add_text(self, name, *args, **kwargs):
        '''
            Overview: add message to text
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['text']
        '''
        assert(name in self._var_names['text'])
        self.logger.add_text(name, *args, **kwargs)

    def add_scalars(self, name, *args, **kwargs):
        '''
            Overview: add message to scalars
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['scalars']
        '''
        assert(name in self._var_names['scalars'])
        self.logger.add_scalars(name, *args, **kwargs)

    def add_histogram(self, name, *args, **kwargs):
        '''
            Overview: add message to histogram
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['histogram']
        '''
        assert(name in self._var_names['histogram'])
        self.logger.add_histogram(name, *args, **kwargs)

    def add_figure(self, name, *args, **kwargs):
        '''
            Overview: add message to figure
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['figure']
        '''
        assert(name in self._var_names['figure'])
        self.logger.add_figure(name, *args, **kwargs)

    def add_image(self, name, *args, **kwargs):
        '''
            Overview: add message to image
            Arguments:
                - name (:obj:`str`): name to add which in self._var_names['image']
        '''
        assert(name in self._var_names['image'])
        self.logger.add_image(name, *args, **kwargs)

    def add_val_list(self, val_list, viz_type):
        '''
            Overview: add val_list info to tb
            Arguments:
                - val_list (:obj:`list`): include element(name, value, step) to be added
                - viz_type (:obs:`str`): must be in ['scalar', 'scalars', 'histogram']
        '''
        assert(viz_type in ['scalar', 'scalars', 'histogram'])
        func_dict = {
            'scalar': self.add_scalar,
            'scalars': self.add_scalars,
            'histogram': self.add_histogram,
        }
        for n, v, s in val_list:
            func_dict[viz_type](n, v, s)

    def _no_contain_name(self, name):
        for k, v in self._var_names.items():
            if name in v:
                return False
        return True

    def register_var(self, name, var_type='scalar'):
        assert(var_type in self._var_names.keys())
        assert(self._no_contain_name(name))
        self._var_names[var_type].append(name)

    @property
    def scalar_var_names(self):
        return self._var_names['scalar']


class VariableRecord(object):
    def __init__(self, length):
        self.var_dict = {'scalar': {}, '1darray': {}}
        self.length = length

    def register_var(self, name, length=None, var_type='scalar'):
        assert(var_type in ['scalar', '1darray'])
        lens = self.length if length is None else length
        self.var_dict[var_type][name] = AverageMeter(lens)

    def update_var(self, info):
        assert(isinstance(info, dict))
        for k, v in info.items():
            var_type = self._get_var_type(k)
            self.var_dict[var_type][k].update(v)

    def _get_var_type(self, k):
        for var_type, var_type_dict in self.var_dict.items():
            if k in var_type_dict.keys():
                return var_type
        raise KeyError("invalid key({}) in variable record".format(k))

    def get_var_names(self, var_type='scalar'):
        return self.var_dict[var_type].keys()

    def get_var_text(self, name, var_type='scalar'):
        assert(var_type in ['scalar', '1darray'])
        if var_type == 'scalar':
            handle_var = self.var_dict[var_type][name]
            return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val, handle_var.avg)
        elif var_type == '1darray':
            return self._get_var_text_1darray(name)

    def get_vars_tb_format(self, keys, cur_step, var_type='scalar', **kwargs):
        assert(var_type in ['scalar', '1darray'])
        if var_type == 'scalar':
            ret = []
            var_keys = self.get_var_names(var_type)
            for k in keys:
                if k in var_keys:
                    v = self.var_dict[var_type][k]
                    ret.append([k, v.avg, cur_step])
            return ret
        elif var_type == '1darray':
            return self._get_vars_tb_format_1darray(keys, cur_step, **kwargs)

    def get_vars_text(self, var_type='scalar'):
        s = '\n'
        count = 0
        for k in self.get_var_names(var_type):
            s += self.get_var_text(k, var_type) + '\t'
            count += 1
            if count % 3 == 0:
                s += '\n'
        s += self._get_vars_text_1darray()
        return s

    def _get_vars_text_1darray(self):
        return ""

    def _get_vars_tb_format_1darray(self, keys, cur_step, **kwargs):
        raise NotImplementedError


class AlphastarVarRecord(VariableRecord):

    # overwrite
    def register_var(self, name, length=None, var_type='scalar', var_item_keys=None):
        assert(var_type in ['scalar', '1darray'])
        lens = self.length if length is None else length
        self.var_dict[var_type][name] = AverageMeter(lens)
        if not hasattr(self, 'var_item_keys'):
            self.var_item_keys = {}
        self.var_item_keys[name] = var_item_keys

    # overwrite
    def _get_vars_text_1darray(self):
        s = "\n"
        for k in self.get_var_names('1darray'):
            val = self.var_dict['1darray'][k].avg
            if k == 'action_type':
                s += '{}:\t'.format(k)
                items = [[n, v] for n, v in zip(self.var_item_keys[k], val) if v > 0]
                items = sorted(items, key=lambda x: float(x[1]), reverse=True)
                for n, v in items:
                    s += '{}({:.2f})  '.format(n, v)
                s += '\n'
            else:
                s += '{}:\t'.format(k)
                for n, v in zip(self.var_item_keys[k], val):
                    s += '{}({:.2f})  '.format(n, v)
                s += '\n'
        return s

    # overwrite
    def _get_vars_tb_format_1darray(self, keys, cur_step, viz_type=None):
        assert(viz_type in ['scalars', 'histogram'])
        if viz_type == 'scalars':
            ret = []
            var_keys = self.get_var_names('1darray')
            for k in keys:
                if k in var_keys:
                    v = self.var_dict['1darray'][k]
                    scalars = {k: v for k, v in zip(self.var_item_keys[k], v.avg)}
                    ret.append([k, scalars, cur_step])
            return ret
        elif viz_type == 'histogram':
            ret = []
            var_keys = self.get_var_names('1darray')
            for k in keys:
                if k in var_keys:
                    v = self.var_dict['1darray'][k]
                    ret.append([k, v.avg, cur_step])
            return ret


class AverageMeter(object):
    """
        Overview: Computes and stores the average and current value, scalar and 1D-array
    """

    def __init__(self, length=0):
        assert(length > 0)
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        assert(isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history, axis=0)


class DistributionTimeImage(object):
    def __init__(self, maxlen=600, val_range=None):
        self.maxlen = maxlen
        self.val_range = val_range
        self.img = np.ones((maxlen, maxlen))
        self.time_step = 0
        self.one_img = np.ones((maxlen, maxlen))

    def add_one_time_step(self, data):
        assert(isinstance(data, np.ndarray))
        data = np.expand_dims(data, 1)
        data = cv2.resize(data, (1, self.maxlen), interpolation=cv2.INTER_LINEAR)
        if self.time_step >= self.maxlen:
            self.img = np.concatenate([self.img[:, 1:], data])
        else:
            self.img[:, self.time_step:self.time_step+1] = data
            self.time_step += 1

    def get_image(self):
        norm_img = np.copy(self.img)
        valid = norm_img[:, :self.time_step]
        if self.val_range is None:
            valid = (valid - valid.min()) / (valid.max() - valid.min())
        else:
            valid = np.clip(valid, self.val_range['min'], self.val_range['max'])
            valid = (valid - self.val_range['min']) / (self.val_range['max'] - self.val_range['min'])
        norm_img[:, :self.time_step] = valid
        return np.stack([self.one_img, norm_img, norm_img], axis=0)
