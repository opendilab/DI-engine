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
import numpy as np
import yaml
from tabulate import tabulate
from tensorboardX import SummaryWriter
from typing import Optional, Tuple, Union, Dict, List, Any

from ding.utils.autolog import TickTime


def build_logger(
    path: str,
    name: Optional[str] = None,
    need_tb: bool = True,
    need_text: bool = True,
    text_level: Union[int, str] = logging.INFO
) -> Tuple[Optional[logging.Logger], Optional['SummaryWriter']]:  # noqa
    r'''
    Overview:
        Build text logger and tensorboard logger.
    Arguments:
        - path (:obj:`str`): Logger(``Textlogger`` & ``SummaryWriter``)'s saved dir
        - name (:obj:`str`): The logger file name
        - need_tb (:obj:`bool`): Whether ``SummaryWriter`` instance would be created and returned
        - need_text (:obj:`bool`): Whether ``loggingLogger`` instance would be created and returned
        - text_level (:obj:`int`` or :obj:`str`): Logging level of ``logging.Logger``, default set to ``logging.INFO``
    Returns:
        - logger (:obj:`Optional[logging.Logger]`): Logger that displays terminal output
        - tb_logger (:obj:`Optional['SummaryWriter']`): Saves output to tfboard, only return when ``need_tb``.
    '''
    if name is None:
        name = 'default'
    logger = LoggerFactory.create_logger(path, name=name) if need_text else None
    tb_name = name + '_tb_logger'
    tb_logger = SummaryWriter(os.path.join(path, tb_name)) if need_tb else None
    return logger, tb_logger


class LoggerFactory(object):

    @classmethod
    def create_logger(cls, path: str, name: str = 'default', level: Union[int, str] = logging.INFO) -> logging.Logger:
        r"""
        Overview:
            Create logger using logging
        Arguments:
            - name (:obj:`str`): Logger's name
            - path (:obj:`str`): Logger's save dir
            - level (:obj:`int` or :obj:`str`): Used to set the level. Reference: ``Logger.setLevel`` method.
        Returns:
            - (:obj:`logging.Logger`): new logging logger
        """
        name += '_logger'
        # ensure the path exists
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        logger = logging.getLogger(name)
        logger_file_path = os.path.join(path, name + '.txt')
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(logger_file_path, 'a')
            fh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
        logger.get_tabulate_vars = LoggerFactory.get_tabulate_vars
        logger.get_tabulate_vars_hor = LoggerFactory.get_tabulate_vars_hor
        return logger

    @staticmethod
    def get_tabulate_vars(variables: Dict[str, Any]) -> str:
        r"""
        Overview:
            Get the text description in tabular form of all vars
        Arguments:
            - variables (:obj:`List[str]`): Names of the vars to query.
        Returns:
            - string (:obj:`str`): Text description in tabular form of all vars
        """
        headers = ["Name", "Value"]
        data = []
        for k, v in variables.items():
            data.append([k, "{:.6f}".format(v)])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s

    @staticmethod
    def get_tabulate_vars_hor(variables: Dict[str, Any]) -> str:
        datak = []
        datav = []
        datak.append("Name")
        datav.append("Value")
        for k, v in variables.items():
            datak.append(k)
            if not isinstance(v, str) and np.isscalar(v):
                datav.append("{:.6f}".format(v))
            else:
                datav.append(v)
        data = [datak, datav]
        s = "\n" + tabulate(data, tablefmt='grid')
        return s


class DistributionTimeImage:
    r"""
    Overview:
        ``DistributionTimeImage`` can be used to store images accorrding to ``time_steps``,
        for data with 3 dims``(time, category, value)``
    Interface:
        ``__init__``, ``add_one_time_step``, ``get_image``
    """

    def __init__(self, maxlen: int = 600, val_range: Optional[dict] = None):
        r"""
        Overview:
            Init the ``DistributionTimeImage`` class
        Arguments:
            - maxlen (:obj:`int`): The max length of data inputs
            - val_range (:obj:`dict` or :obj:`None`): Dict with ``val_range['min']`` and ``val_range['max']``.
        """
        self.maxlen = maxlen
        self.val_range = val_range
        self.img = np.ones((maxlen, maxlen))
        self.time_step = 0
        self.one_img = np.ones((maxlen, maxlen))

    def add_one_time_step(self, data: np.ndarray) -> None:
        r"""
        Overview:
            Step one timestep in ``DistributionTimeImage`` and add the data to distribution image
        Arguments:
            - data (:obj:`np.ndarray`): The data input
        """
        assert (isinstance(data, np.ndarray))
        data = np.expand_dims(data, 1)
        data = np.resize(data, (1, self.maxlen))
        if self.time_step >= self.maxlen:
            self.img = np.concatenate([self.img[:, 1:], data])
        else:
            self.img[:, self.time_step:self.time_step + 1] = data
            self.time_step += 1

    def get_image(self) -> np.ndarray:
        r"""
        Overview:
            Return the distribution image
        Returns:
            - img (:obj:`np.ndarray`): The calculated distribution image
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
        - result (:obj:`dict`): The result to print
        - direct_print (:obj:`bool`): Whether to print directly
    Returns:
        - string (:obj:`str`): The pretty-printed result in str format
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
