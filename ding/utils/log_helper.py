import json
from ditk import logging
import os
from typing import Optional, Tuple, Union, Dict, Any

import ditk.logging
import numpy as np
import yaml
from hbutils.system import touch
from tabulate import tabulate

from .log_writer_helper import DistributedWriter


def build_logger(
    path: str,
    name: Optional[str] = None,
    need_tb: bool = True,
    need_text: bool = True,
    text_level: Union[int, str] = logging.INFO
) -> Tuple[Optional[logging.Logger], Optional['SummaryWriter']]:  # noqa
    r"""
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
    """
    if name is None:
        name = 'default'
    logger = LoggerFactory.create_logger(path, name=name) if need_text else None
    tb_name = name + '_tb_logger'
    tb_logger = TBLoggerFactory.create_logger(os.path.join(path, tb_name)) if need_tb else None
    return logger, tb_logger


class TBLoggerFactory(object):
    tb_loggers = {}

    @classmethod
    def create_logger(cls: type, logdir: str) -> DistributedWriter:
        if logdir in cls.tb_loggers:
            return cls.tb_loggers[logdir]
        tb_logger = DistributedWriter(logdir)
        cls.tb_loggers[logdir] = tb_logger
        return tb_logger


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
        ditk.logging.try_init_root(level)

        logger_name = f'{name}_logger'
        logger_file_path = os.path.join(path, f'{logger_name}.txt')
        touch(logger_file_path)

        logger = ditk.logging.getLogger(logger_name, level, [logger_file_path])
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
        column_to_divide = 5  # which includes the header "Name & Value"

        datak = []
        datav = []

        divide_count = 0
        for k, v in variables.items():
            if divide_count == 0 or divide_count >= (column_to_divide - 1):
                datak.append("Name")
                datav.append("Value")
                if divide_count >= (column_to_divide - 1):
                    divide_count = 0
            divide_count += 1

            datak.append(k)
            if not isinstance(v, str) and np.isscalar(v):
                datav.append("{:.6f}".format(v))
            else:
                datav.append(v)

        s = "\n"
        row_number = len(datak) // column_to_divide + 1
        for row_id in range(row_number):
            item_start = row_id * column_to_divide
            item_end = (row_id + 1) * column_to_divide
            if (row_id + 1) * column_to_divide > len(datak):
                item_end = len(datak)
            data = [datak[item_start:item_end], datav[item_start:item_end]]
            s = s + tabulate(data, tablefmt='grid') + "\n"

        return s


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
