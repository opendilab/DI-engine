import logging
import numpy as np
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
        self._scalar_var_names = []
        self._text_var_names = []
        self._scalars_var_names = []

    def add_scalar(self, name, val, step):
        assert(name in self._scalar_var_names)
        self.logger.add_scalar(name, val, step)

    def add_text(self, name, text, step):
        assert(name in self._text_var_names)
        self.logger.add_text(name, text, step)

    def add_scalars(self, name, val, step):
        assert(name in self._scalars_var_names)
        self.logger.add_scalars(name, val, step)

    def add_scalar_list(self, scalar_list):
        for n, v, s in scalar_list:
            self.add_scalar(n, v, s)

    def register_var(self, name, var_type='scalar'):
        assert(var_type in ['scalar', 'text', 'scalars'])
        assert(name not in self._scalar_var_names)
        assert(name not in self._text_var_names)
        if var_type == 'scalar':
            self._scalar_var_names.append(name)
        elif var_type == 'text':
            self._text_var_names.append(name)
        elif var_type == 'scalars':
            self._scalars_var_names.append(name)

    @property
    def scalar_var_names(self):
        return self._scalar_var_names


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
