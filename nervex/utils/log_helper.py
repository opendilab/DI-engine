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

from nervex.utils.autolog import TickTime


def build_logger(
        path: str,
        name: Optional[str] = None,
        need_tb: bool = True,
        need_text: bool = True,
        text_level: Union[int, str] = logging.INFO
) -> Tuple[Optional['TextLogger'], Optional['SummaryWriter']]:  # noqa
    r'''
    Overview:
        Build TextLogger and SummaryWriter.
    Arguments:
        - path (:obj:``str``): logger(Textlogger & SummaryWriter)'s saved dir
        - name (:obj:``str``): the logger file name
        - need_tb (:obj:``bool``): whether SummaryWriter instance would be created and returned
        - need_text (:obj:``bool``): whether TextLogger instance would be created and returned
        - text_level (:obj:``int`` or :obj:``str``): logging level of TextLogger, default set to ``logging.INFO``
    Returns:
        - logger (:obj:``Optional['TextLogger']``): logger that displays terminal output
        - tb_logger (:obj:``Optional['SummaryWriter']``): logger that saves output to tensorboard, \
            only return when ``need_tb`` is True
    '''
    if name is None:
        name = 'default'
    logger = TextLogger(path, name=name) if need_text else None
    tb_name = name + '_tb_logger'
    tb_logger = SummaryWriter(os.path.join(path, tb_name)) if need_tb else None
    return logger, tb_logger


class TextLogger(object):
    r"""
    Overview:
        Logger that saves terminal output to file
    Interface:
        __init__, info, debug
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

    def print_vars(self, vars: Dict[str, Any], level: int = logging.INFO) -> None:
        r"""
        Overview:
            Get the text description in tabular form of all vars
        Arguments:
            - names (:obj:`List[str]`): names of the vars to query. If you want to query all vars, you can omit this \
                argument and thus ``need_print`` will be set to True all the time by default.
            - var_type (:obj:`str`): default set to scalar, support ['scalar']
            - level (:obj:`int`): log level
        Returns:
            - ret (:obj:`list` of :obj:`str`): text description in tabular form of all vars
        """
        headers = ["Name", "Value"]
        data = []
        for k, v in vars.items():
            data.append([k, "{:.6f}".format(v)])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        if level >= logging.INFO:
            self.info(s)

    def print_vars_hor(self, vars: Dict[str, Any], level: int = logging.INFO) -> None:
        datak = []
        datav = []
        datak.append("Name")
        datav.append("Value")
        for k, v in vars.items():
            datak.append(k)
            datav.append("{:.6f}".format(v))
        data = [datak, datav]
        s = "\n" + tabulate(data, tablefmt='grid')
        if level >= logging.INFO:
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

    def debug(self, s: str) -> None:
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

    @property
    def level(self) -> int:
        return self.logger.level


class DistributionTimeImage:
    r"""
    Overview:
        DistributionTimeImage can be used to store images accorrding to time_steps,
        for data with 3 dims(time, category, value)
    Interface:
        __init__, add_one_time_step, get_image
    """

    def __init__(self, maxlen: int = 600, val_range: Optional[dict] = None):
        r"""
        Overview:
            init the DistributionTimeImage class
        Arguments:
            - maxlen (:obj:`int`): the max length of data inputs
            - val_range (:obj:`dict` or :obj:`None`): contain :obj:`int` type val_range['min'] and val_range['max'], \
                default set to None
        """
        self.maxlen = maxlen
        self.val_range = val_range
        self.img = np.ones((maxlen, maxlen))
        self.time_step = 0
        self.one_img = np.ones((maxlen, maxlen))

    def add_one_time_step(self, data: np.ndarray) -> None:
        r"""
        Overview:
            step one timestep in DistributionTimeImage and add the data to distribution image
        Arguments:
            - data (:obj:`np.ndarray`):the data input
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
            return the distribution image
        Returns:
            - img (:obj:`np.ndarray`): the calculated distribution image
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
