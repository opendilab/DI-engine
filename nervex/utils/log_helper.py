'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. log helper, used to help to save logger on terminal, tensorboard or save file.
    2. CountVar, to help counting number.
'''
import json
import logging
import numbers
import os
import cv2
import numpy as np
import yaml
from tabulate import tabulate
from tensorboardX import SummaryWriter
from typing import Optional, Tuple, Union, Dict, List, Any

from nervex.utils.autolog import TickTime


def build_logger(
        path: str,
        name: Optional[str] = None,
        need_tb: bool = False,
        text_level: Union[int, str] = logging.INFO
) -> Tuple['TextLogger', Optional['TensorBoardLogger']]:  # noqa
    r'''
    Overview:
        Build TextLogger, if needed can also build VariableRecord and TensorBoardLogger.
        TextLogger and TensorboardLogger are real log files, VariableRecord is used to store variables and often
        outputed to TextLogger in tabular form.
    Arguments:
        - path (:obj:`str`): logger(Textlogger & TensorBoardLogger)'s saved dir
        - name (:obj:`str`): the logger file name
        - need_tb (:obj:`bool`): whether variable record instance would be returned
        - text_level (:obj:`int` or :obj:`str`): logging level of TextLogger, default set to ``logging.INFO``
    Returns:
        - logger (:obj:`TextLogger`): logger that displays terminal output
        - tb_logger (:obj:`TensorBoardLogger`): logger that saves output to tensorboard, \
            only return when ``need_tb`` is True
    '''
    logger = TextLogger(path, name=name)
    tb_logger = TensorBoardLogger(path, name=name) if need_tb else None
    # variable_record = VariableRecord(var_len) if need_var else None
    return logger, tb_logger


def get_default_logger(name: Optional[str] = None) -> logging.Logger:
    r"""
    Overview:
        Get the logger using logging.getLogger
    Arguments:
        - name (:obj:`str`): the name of logger, if None then get 'default_logger'
    Notes:
        you can reference Logger.manager.getLogger(name) in the python3 /logging/__init__.py
    """
    if name is None:
        name = 'default_logger'
    return logging.getLogger(name)


class TextLogger(object):
    r"""
    Overview:
        Logger that saves terminal output to file
    Interface:
        __init__, info
    """

    def __init__(self, path: str, name: str = 'default', level: Union[int, str] = logging.INFO) -> None:
        r"""
        Overview:
            initialization method, create logger.
        Arguments:
            - path (:obj:`str`): logger's save dir
            - name (:obj:`str`): logger's name, default set to 'default'
            - level (:obj:`int` or :obj:`str`): Set the logging level of logger, reference Logger class setLevel method.
        """
        name += '_logger'
        # ensure the path exists
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        self.logger = self._create_logger(name, os.path.join(path, name + '.txt'), level=level)

    def _create_logger(self, name: str, path: str, level: Union[int, str] = logging.INFO) -> logging.Logger:
        r"""
        Overview:
            Create logger using logging
        Arguments:
            - name (:obj:`str`): logger's name
            - path (:obj:`str`): logger's save dir
            - level (:obj:`int` or :obj:`str`): used to set the logging level of logger, you can reference \
                Logger class ``setLevel`` method.
        Returns:
            - (:obj`logging.Logger`): new logging logger
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(path, 'a')
            fh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
        return logger

    def print_vars(self, vars: Dict[str, Any]) -> None:
        r"""
        Overview:
            Get the text description in tabular form of all vars
        Arguments:
            - names (:obj:`List[str]`): names of the vars to query. If you want to query all vars, you can omit this \
                argument and thus ``need_print`` will be set to True all the time by default.
            - var_type (:obj:`str`): default set to scalar, support ['scalar']
        Returns:
            - ret (:obj:`list` of :obj:`str`): text description in tabular form of all vars
        """
        headers = ["Name", "Avg"]
        data = []
        for k, v in vars.items():
            data.append([k, "{:.6f}".format(v)])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        self.info(s)

    def info(self, s: str) -> None:
        r"""
        Overview:
            add message to logger
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.info(s)

    def bug(self, s: str) -> None:
        r"""
        Overview:
            call logger.debug
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.debug(s)

    def error(self, s: str) -> None:
        self.logger.error(s)


# class TensorBoardLogger(object):
#     r"""
#     Overview:
#         logger that save message to tensorboard
#     Interface:
#         __init__, add_scalar, add_text, add_scalars, add_histogram, add_figure, add_image, add_scalar_list,
#         register_var, scalar_var_names
#     """

#     def __init__(self, path: str, name: str = 'default') -> None:
#         r"""
#         Overview:
#             initialization method, create logger and set var names.
#         Arguments:
#             - path (:obj:`str`): logger save dir
#             - name (:obj:`str`): logger name, default set to 'default'
#         """
#         name += '_tb_logger'
#         self.logger = SummaryWriter(os.path.join(path, name))  # get summary writer
#         self._var_names = {
#             'scalar': [],
#             'text': [],
#             'scalars': [],
#             'histogram': [],
#             'figure': [],
#             'image': [],
#         }

#     def print_vars(self, vars: Dict[str, Any], cur_step: int, **kwargs) -> List[list]:
#         r"""
#         Overview:
#             Get the var dict and print it to tensorboard
#         Arguments:
#             - vars (:obj:`Dict[str, Any]`): vars dict containing name and its value
#             - cur_step (:obj:`int`): the current step
#         Returns:
#             - ret (:obj:`List[list]`): a list containing vars length lists, each containing one var's \
#                 tb_format tuple(name, value, step)
#         """

#         ret = []
#         for k, v in vars.items():
#             self.add_scalar(k, v, cur_step)
#         #     ret.append([k, v, cur_step])
#         # self.add_val_list(ret, 'scalar')

#     def add_scalar(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add message to scalar
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['scalar']
#         """
#         # assert (name in self._var_names['scalar'])
#         self.logger.add_scalar(name, *args, **kwargs)

#     def add_text(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add message to text
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['text']
#         """
#         assert (name in self._var_names['text'])
#         self.logger.add_text(name, *args, **kwargs)

#     def add_scalars(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add messages to scalar
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['scalars']
#         """
#         assert (name in self._var_names['scalars'])
#         self.logger.add_scalars(name, *args, **kwargs)

#     def add_histogram(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add message to histogram
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['histogram']
#         """
#         assert (name in self._var_names['histogram'])
#         self.logger.add_histogram(name, *args, **kwargs)

#     def add_figure(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add message to figure
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['figure']
#         """
#         assert (name in self._var_names['figure'])
#         self.logger.add_figure(name, *args, **kwargs)

#     def add_image(self, name: str, *args, **kwargs) -> None:
#         r"""
#         Overview:
#             add message to image
#         Arguments:
#             - name (:obj:`str`): name to add which in self._var_names['image']
#         """
#         assert (name in self._var_names['image'])
#         self.logger.add_image(name, *args, **kwargs)

#     def add_val_list(self, val_list: list, viz_type: str) -> None:
#         r"""
#         Overview:
#             add val_list info to tb
#         Arguments:
#             - val_list (:obj:`list`): include element(name, value, step) to be added
#             - viz_type (:obs:`str`): must be in ['scalar', 'scalars', 'histogram']
#         """
#         assert (viz_type in ['scalar', 'scalars', 'histogram'])
#         func_dict = {
#             'scalar': self.add_scalar,
#             'scalars': self.add_scalars,
#             'histogram': self.add_histogram,
#         }
#         for n, v, s in val_list:  # name, value, step
#             func_dict[viz_type](n, v, s)

#     def _no_contain_name(self, name: str) -> bool:
#         for k, v in self._var_names.items():
#             if name in v:
#                 return False
#         return True

#     def register_var(self, name: str, var_type: str = 'scalar') -> None:
#         r"""
#         Overview:
#             Add var to self._var_names
#         Arguments:
#             - name (:obj:`str`): name to add
#             - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar', \
#                 support [scalar', 'text', 'scalars', 'histogram', 'figure', 'image']
#         """
#         assert (var_type in self._var_names.keys())
#         assert (self._no_contain_name(name))
#         self._var_names[var_type].append(name)

#     @property
#     def scalar_var_names(self):
#         r"""
#         Overview:
#             return scalar_var_names
#         Returns:
#             - names(:obj:`list` of :obj:`str`): self._var_names['scalar']
#         """
#         return self._var_names['scalar']


class TensorBoardLogger(object):
    r"""
    Overview:
        logger that save message to tensorboard
    Interface:
        __init__, add_scalar, add_text, add_scalars, add_histogram, add_figure, add_image, add_scalar_list,
        register_var, scalar_var_names
    """

    def __init__(self, path: str, name: str = 'default') -> None:
        r"""
        Overview:
            initialization method, create logger and set var names.
        Arguments:
            - path (:obj:`str`): logger save dir
            - name (:obj:`str`): logger name, default set to 'default'
        """
        name += '_tb_logger'
        self.logger = SummaryWriter(os.path.join(path, name))  # get summary writer
        self._var_names = {
            'scalar': [],
            'text': [],
            'scalars': [],
            'histogram': [],
            'figure': [],
            'image': [],
        }

    def print_vars(self, vars: Dict[str, Any], cur_step: int, viz_type: str = 'scalar', **kwargs) -> List[list]:
        r"""
        Overview:
            Get the var dict and print it to tensorboard
        Arguments:
            - vars (:obj:`Dict[str, Any]`): vars dict containing name and its value
            - cur_step (:obj:`int`): the current step
            - viz_type (:obs:`str`): must be in ['scalar', 'scalars', 'histogram'], default set to 'scalar'
        Returns:
            - ret (:obj:`List[list]`): a list containing vars length lists, each containing one var's \
                tb_format tuple(name, value, step)
        """
        assert (viz_type in ['scalar', 'scalars', 'histogram'])
        func_dict = {
            'scalar': self.add_scalar,
            'scalars': self.add_scalars,
            'histogram': self.add_histogram,
        }
        ret = []
        for k, v in vars.items():
            func_dict[viz_type](k, v, cur_step)

    def add_scalar(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add message to scalar
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['scalar']
        """
        assert name in self._var_names['scalar'], name
        self.logger.add_scalar(name, *args, **kwargs)

    def add_text(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add message to text
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['text']
        """
        assert (name in self._var_names['text'])
        self.logger.add_text(name, *args, **kwargs)

    def add_scalars(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add messages to scalar
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['scalars']
        """
        assert (name in self._var_names['scalars'])
        self.logger.add_scalars(name, *args, **kwargs)

    def add_histogram(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add message to histogram
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['histogram']
        """
        assert (name in self._var_names['histogram'])
        self.logger.add_histogram(name, *args, **kwargs)

    def add_figure(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add message to figure
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['figure']
        """
        assert (name in self._var_names['figure'])
        self.logger.add_figure(name, *args, **kwargs)

    def add_image(self, name: str, *args, **kwargs) -> None:
        r"""
        Overview:
            add message to image
        Arguments:
            - name (:obj:`str`): name to add which in self._var_names['image']
        """
        assert (name in self._var_names['image'])
        self.logger.add_image(name, *args, **kwargs)

    def add_val_list(self, val_list: list, viz_type: str) -> None:
        r"""
        Overview:
            add val_list info to tb
        Arguments:
            - val_list (:obj:`list`): include element(name, value, step) to be added
            - viz_type (:obs:`str`): must be in ['scalar', 'scalars', 'histogram']
        """
        assert (viz_type in ['scalar', 'scalars', 'histogram'])
        func_dict = {
            'scalar': self.add_scalar,
            'scalars': self.add_scalars,
            'histogram': self.add_histogram,
        }
        for n, v, s in val_list:  # name, value, step
            func_dict[viz_type](n, v, s)

    def _no_contain_name(self, name: str) -> bool:
        for k, v in self._var_names.items():
            if name in v:
                return False
        return True

    def register_var(self, name: str, var_type: str = 'scalar') -> None:
        r"""
        Overview:
            Add var to self._var_names
        Arguments:
            - name (:obj:`str`): name to add
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar', \
                support [scalar', 'text', 'scalars', 'histogram', 'figure', 'image']
        """
        assert (var_type in self._var_names.keys())
        assert (self._no_contain_name(name))
        self._var_names[var_type].append(name)

    @property
    def scalar_var_names(self):
        r"""
        Overview:
            return scalar_var_names
        Returns:
            - names(:obj:`list` of :obj:`str`): self._var_names['scalar']
        """
        return self._var_names['scalar']


# This version contains LoggedModel subclass instance(es) to replace AverageMeter, used to monitor variables
# class VariableRecord(object):
#     r"""
#     Overview:
#         Records and manages variables using ``LoggedModel``. Then output variable info in tensorboard and text format,
#         which can be used by ``TextLogger`` and ``TensorBoardLogger``.
#     Interface:
#         __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
#     """

#     def __init__(self, monitors: Union['LoggedModel', List['LoggedModel']]) -> None:  # noqa
#         r"""
#         Overview:
#             init the VariableRecord
#         Arguments:
#             - monitor (:obj:`Union[LoggedModel, List[LoggedModel']]): one single or a list of LoggedModel subclass \
#                 instance(es), used to record and manage all var's history ``(duration, value)`` list.
#         """
#         # self.var_dict = {'scalar': {}}
#         if not isinstance(monitors, list):
#             monitors = [monitors]
#         self.monitors = monitors

#     def _tick_monitor_step(self) -> None:
#         for monitor in self.monitors:
#             if isinstance(monitor.time, TickTime):
#                 monitor.time.step()

#     def _get_attributes(self, property_name: str) -> List[str]:
#         attributes = []
#         for monitor in self.monitors:
#             attrs = monitor.get_attrs(property_name)  # todo: LoggedModel's get_attrs
#             if isinstance(attrs, list):
#                 attributes += attrs
#             else:
#                 attributes.append(attrs)
#         return attributes

#     def update_var(self, info: Dict[str, Any]) -> None:
#         r"""
#         Overview:
#             Update vars in ``self.monitors`` according to ``info`` dict containing var name and var new value.
#             For tick_monitor, it should call ``step`` to make time tick once.
#         Arguments:
#             - info (:obj:`Dict[str, Any]`): key is var name and value is the corresponding value to be updated
#         """
#         assert isinstance(info, dict)
#         for k, v in info.items():
#             attributes = self._get_attributes(k)

#             var_type = self._get_var_type(k)
#             self.var_dict[var_type][k].update(v)

#     def _get_var_type(self, k: str) -> str:
#         r"""
#         Overview:
#             Find a variable's type according to its name ``k``
#         Arguments:
#             - k (:obj:`str`): a variable's name
#         Returns:
#             - var_type (:obj:`str`): the variable's type in str format
#         """
#         for var_type, var_type_dict in self.var_dict.items():
#             if k in var_type_dict.keys():
#                 return var_type
#         raise KeyError("invalid key({}) in variable record".format(k))

#     def get_var_names(self, var_type: str = 'scalar') -> List[str]:
#         r"""
#         Overview:
#             Get the corresponding variable names of a certain var_type
#         Arguments:
#             - var_type (:obj:`str`): defalut set to 'scalar', support [scalar']
#         Returns:
#             - keys (:obj:`List[str]`): the var name list of the certain ``var_type``
#         """
#         return self.var_dict[var_type].keys()

#     def get_var_text(self, name: str, var_type='scalar') -> str:
#         r"""
#         Overview:
#             Get the text discription of one single variable
#         Arguments:
#             - name (:obj:`str`): name of the var to query
#             - var_type (:obj:`str`): default set to scalar, support ['scalar']
#         Returns:
#             - text(:obj:`str`): the corresponding text description of the variable
#         """
#         assert (var_type in ['scalar'])
#         if var_type == 'scalar':
#             handle_var = self.var_dict[var_type][name]
#             return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val, handle_var.avg)
#         else:
#             raise NotImplementedError

#     def get_vars_text(self, names: Optional[List[str]] = None, var_type='scalar') -> str:
#         r"""
#         Overview:
#             Get the text description in tabular form of all vars
#         Arguments:
#             - names (:obj:`List[str]`): names of the vars to query. If you want to query all vars, you can omit this \
#                 argument and thus ``need_print`` will be set to True all the time by default.
#             - var_type (:obj:`str`): default set to scalar, support ['scalar']
#         Returns:
#             - ret (:obj:`list` of :obj:`str`): text description in tabular form of all vars
#         """
#         headers = ["Name", "Value", "Avg"]
#         data = []
#         if var_type == 'scalar':
#             for k in self.get_var_names('scalar'):
#                 need_print = True if names is None else k in names
#                 if need_print:
#                     handle_var = self.var_dict['scalar'][k]
#                     data.append([k, "{:.6f}".format(handle_var.val), "{:.6f}".format(handle_var.avg)])
#             s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
#             return s
#         else:
#             raise NotImplementedError

#     def get_vars_tb_format(self, keys: List[str], cur_step: int, var_type: str = 'scalar', **kwargs) -> List[list]:
#         r"""
#         Overview:
#             Get the tensorboard_format description of var
#         Arguments:
#             - keys (:obj:`list` of :obj:`str`): keys(names) of the var to query
#             - cur_step (:obj:`int`): the current step
#             - var_type(:obj:`str`): default set to scalar, support support ['scalar']
#         Returns:
#             - ret (:obj:`List[list]`): a list containing keys length lists, each containing corresponding var's \
#                 tb_format info
#         """
#         assert (var_type in ['scalar'])
#         if var_type == 'scalar':
#             ret = []
#             var_keys = self.get_var_names(var_type)
#             for k in keys:
#                 if k in var_keys:
#                     v = self.var_dict[var_type][k]
#                     ret.append([k, v.avg, cur_step])
#             return ret
#         else:
#             raise NotImplementedError

# class VariableRecord(object):
#     r"""
#     Overview:
#         A logger that records variable for further process, now supports type ['scalar'].
#         For each scalar var, use ``LoggedModel`` to record and manage.
#     Interface:
#         __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
#     """

#     def __init__(self, length: int) -> None:
#         r"""
#         Overview:
#             init the VariableRecord
#         Arguments:
#             - length (:obj:`int`): the length to average across, if less than 10 then will be set to 10
#         """
#         self.var_dict = {'scalar': {}}
#         self.length = max(length, 10)  # at least average across 10 iteration

#     def register_var(self, name: str, length: Optional[int] = None, var_type: str = 'scalar') -> None:
#         r"""
#         Overview:
#             add var to self_var._names, calculate it's average value
#         Arguments:
#             - name (:obj:`str`): name to add
#             - length (:obj:`int` or :obj:`None`): length of iters to average, default set to self.length
#             - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar'
#         """
#         assert (var_type in ['scalar'])
#         lens = self.length if length is None else length
#         self.var_dict[var_type][name] = AverageMeter(lens)  # todo

#     def update_var(self, info: Dict[str, Any]) -> None:
#         r"""
#         Overview:
#             Update vars according to ``info`` dict containing var name and var new value
#         Arguments:
#             - info (:obj:`Dict[str, Any]`): key is var name and value is the corresponding value to be updated
#         """
#         assert isinstance(info, dict)
#         for k, v in info.items():
#             var_type = self._get_var_type(k)
#             self.var_dict[var_type][k].update(v)

#     def _get_var_type(self, k: str) -> str:
#         r"""
#         Overview:
#             Find a variable's type according to its name ``k``
#         Arguments:
#             - k (:obj:`str`): a variable's name
#         Returns:
#             - var_type (:obj:`str`): the variable's type in str format
#         """
#         for var_type, var_type_dict in self.var_dict.items():
#             if k in var_type_dict.keys():
#                 return var_type
#         raise KeyError("invalid key({}) in variable record".format(k))

#     def get_var_names(self, var_type: str = 'scalar') -> List[str]:
#         r"""
#         Overview:
#             Get the corresponding variable names of a certain var_type
#         Arguments:
#             - var_type (:obj:`str`): defalut set to 'scalar', support [scalar']
#         Returns:
#             - keys (:obj:`List[str]`): the var name list of the certain ``var_type``
#         """
#         return self.var_dict[var_type].keys()

#     def get_var_text(self, name: str, var_type='scalar') -> str:
#         r"""
#         Overview:
#             Get the text discription of one single variable
#         Arguments:
#             - name (:obj:`str`): name of the var to query
#             - var_type (:obj:`str`): default set to scalar, support ['scalar']
#         Returns:
#             - text(:obj:`str`): the corresponding text description of the variable
#         """
#         assert (var_type in ['scalar'])
#         if var_type == 'scalar':
#             handle_var = self.var_dict[var_type][name]
#             return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val, handle_var.avg)
#         else:
#             raise NotImplementedError

#     def get_vars_text(self, names: Optional[List[str]] = None, var_type='scalar') -> str:
#         r"""
#         Overview:
#             Get the text description in tabular form of all vars
#         Arguments:
#             - names (:obj:`List[str]`): names of the vars to query. If you want to query all vars, you can omit this \
#                 argument and thus ``need_print`` will be set to True all the time by default.
#             - var_type (:obj:`str`): default set to scalar, support ['scalar']
#         Returns:
#             - ret (:obj:`list` of :obj:`str`): text description in tabular form of all vars
#         """
#         headers = ["Name", "Value", "Avg"]
#         data = []
#         if var_type == 'scalar':
#             for k in self.get_var_names('scalar'):
#                 need_print = True if names is None else k in names
#                 if need_print:
#                     handle_var = self.var_dict['scalar'][k]
#                     data.append([k, "{:.6f}".format(handle_var.val), "{:.6f}".format(handle_var.avg)])
#             s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
#             return s
#         else:
#             raise NotImplementedError

#     def get_vars_tb_format(self, keys: List[str], cur_step: int, var_type: str = 'scalar', **kwargs) -> List[list]:
#         r"""
#         Overview:
#             Get the tensorboard_format description of var
#         Arguments:
#             - keys (:obj:`list` of :obj:`str`): keys(names) of the var to query
#             - cur_step (:obj:`int`): the current step
#             - var_type(:obj:`str`): default set to scalar, support support ['scalar']
#         Returns:
#             - ret (:obj:`List[list]`): a list containing keys length lists, each containing corresponding var's \
#                 tb_format info
#         """
#         assert (var_type in ['scalar'])
#         if var_type == 'scalar':
#             ret = []
#             var_keys = self.get_var_names(var_type)
#             for k in keys:
#                 if k in var_keys:
#                     v = self.var_dict[var_type][k]
#                     ret.append([k, v.avg, cur_step])
#             return ret
#         else:
#             raise NotImplementedError

# class AverageMeter(object):
#     r"""
#     Overview:
#         Computes and stores the average and current value, scalar and 1D-array
#     Interface:
#         __init__, reset, update
#     """

#     def __init__(self, length=0):
#         r"""
#         Overview:
#             init AverageMeter class
#         Arguments:
#             - length (:obj:`int`) : set the default length of iters to average
#         """
#         assert (length > 0)
#         self.length = length
#         self.reset()

#     def reset(self):
#         r"""
#         Overview:
#             reset AverageMeter class
#         """
#         self.history = []
#         self.val = 0.0
#         self.avg = 0.0

#     def update(self, val):
#         r"""
#         Overview:
#             update AverageMeter class, append the val to the history and calculate the average
#         Arguments:
#             - val (:obj:`numbers.Integral` or :obj:`list` or :obj:`numbers.Real`): the latest value
#         """
#         assert (isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
#         self.history.append(val)
#         if len(self.history) > self.length:
#             del self.history[0]

#         self.val = self.history[-1]
#         self.avg = np.mean(self.history, axis=0)


class DistributionTimeImage(object):
    r"""
    Overview:
        DistributionTimeImage can be used to store images accorrding to time_steps,
        for data with 3 dims(time, category, value)
    Interface:
        __init__, add_one_time_step, get_image
    """

    def __init__(self, maxlen=600, val_range=None):
        r"""
        Overview:
            init the DistributionTimeImage class
        Arguments:
            - maxlen (:obj:`int`): the max length of data inputs
            - val_range (:obj:`dict` or :obj:`None`): contain :obj:`int` type val_range['min'] and val_range['max'],
                                                      default set to None
        """
        self.maxlen = maxlen
        self.val_range = val_range
        self.img = np.ones((maxlen, maxlen))
        self.time_step = 0
        self.one_img = np.ones((maxlen, maxlen))

    def add_one_time_step(self, data):
        r"""
        Overview:
            step one timestep in DistributionTimeImage and add the data to distribution image
        Arguments:
            - data (:obj:`np.array`):the data input
        """
        assert (isinstance(data, np.ndarray))
        data = np.expand_dims(data, 1)
        data = cv2.resize(data, (1, self.maxlen), interpolation=cv2.INTER_LINEAR)
        if self.time_step >= self.maxlen:
            self.img = np.concatenate([self.img[:, 1:], data])
        else:
            self.img[:, self.time_step:self.time_step + 1] = data
            self.time_step += 1

    def get_image(self):
        r"""
        Overview:
            return the distribution image
        Returns:
            img (:obj:`bp.ndarray`): the calculated distribution image
        """
        norm_img = np.copy(self.img)
        valid = norm_img[:, :self.time_step]
        if self.val_range is None:
            valid = (valid - valid.min()) / (valid.max() - valid.min())
        else:
            valid = np.clip(valid, self.val_range['min'], self.val_range['max'])
            valid = (valid - self.val_range['min']) / (self.val_range['max'] - self.val_range['min'])
        norm_img[:, :self.time_step] = valid
        return np.stack([self.one_img, norm_img, norm_img], axis=0)


def pretty_print(result: dict, direct_print: bool = True) -> str:
    r"""
    Overview:
        Print a dict ``result`` in a pretty way
    Arguments:
        - result (:obj:`dict`): the result to print
        - direct_print (:obj:`bool`): whether to print directly
    Returns:
        - string (:obj:`str`): the pretty-printed result in str format
    """
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    string = yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
    if direct_print:
        print(string)
    return string
