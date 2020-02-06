import logging
import numpy as np
import cv2
import os
from tensorboardX import SummaryWriter


def build_logger(cfg, name=None, rank=0):
    '''
        Note: Only support rank0 logger
    '''
    if rank == 0:
        path = cfg.common.save_path
        logger = TextLogger(path, name=name)
        tb_logger = TensorBoardLogger(path, name=name)
        scalar_record = ScalarRecord(cfg.logger.print_freq)
        return logger, tb_logger, scalar_record
    else:
        return None, None, None


class TextLogger(object):
    def __init__(self, path, name=None):
        if name is None:
            name = 'default_logger'
        self.logger = self._create_logger(name, os.path.join(path, name+'.txt'))

    def _create_logger(self, name, path, level=logging.INFO):
        logger = logging.getLogger(name)
        if not logger.handlers:
            formatter = logging.Formatter(
                '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(path)
            fh.setFormatter(formatter)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger

    def info(self, s):
        self.logger.info(s)


class TensorBoardLogger(object):
    def __init__(self, path, name=None):
        if name is None:
            name = 'default_tb_logger'
        self.logger = SummaryWriter(os.path.join(path, name))
        self._var_names = {
            'scalar': [],
            'text': [],
            'scalars': [],
            'histogram': [],
            'figure': [],
            'image': [],
        }

    def add_scalar(self, name, *args, **kwargs):
        assert(name in self._var_names['scalar'])
        self.logger.add_scalar(name, *args, **kwargs)

    def add_text(self, name, *args, **kwargs):
        assert(name in self._var_names['text'])
        self.logger.add_text(name, *args, **kwargs)

    def add_scalars(self, name, *args, **kwargs):
        assert(name in self._var_names['scalars'])
        self.logger.add_scalars(name, *args, **kwargs)

    def add_histogram(self, name, *args, **kwargs):
        assert(name in self._var_names['histogram'])
        self.logger.add_histogram(name, *args, **kwargs)

    def add_figure(self, name, *args, **kwargs):
        assert(name in self._var_names['figure'])
        self.logger.add_figure(name, *args, **kwargs)

    def add_image(self, name, *args, **kwargs):
        assert(name in self._var_names['image'])
        self.logger.add_image(name, *args, **kwargs)

    def add_scalar_list(self, scalar_list):
        for n, v, s in scalar_list:
            self.add_scalar(n, v, s)

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


class ScalarRecord(object):
    def __init__(self, length):
        self.var_dict = {}
        self.length = length

    def register_var(self, name, length=None):
        lens = self.length if length is None else length
        self.var_dict[name] = ScalarAverageMeter(lens)

    def update_var(self, info):
        assert(isinstance(info, dict))
        for k, v in info.items():
            self.var_dict[k].update(v)

    def get_var_names(self):
        return self.var_dict.keys()

    def get_var(self, name):
        handle_var = self.var_dict[name]
        return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val, handle_var.avg)

    def get_var_tb_format(self, keys, cur_step):
        ret = []
        for k in keys:
            if k in self.var_dict.keys():
                v = self.var_dict[k]
                ret.append([k, v.avg, cur_step])
        return ret

    def get_var_all(self):
        s = '\n'
        count = 0
        for k in self.get_var_names():
            s += self.get_var(k) + '\t'
            count += 1
            if count % 3 == 0:
                s += '\n'
        return s


class ScalarAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sums = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sums += val*num
            self.count += num
            self.avg = self.sums / self.count


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
