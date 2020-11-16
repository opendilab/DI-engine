"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
import time
from typing import Callable

import torch
from easydict import EasyDict


def build_time_helper(cfg: EasyDict = None, wrapper_type: str = None) -> Callable[[], 'TimeWrapper']:
    r"""
    Overview:
        build the timehelper

    Arguments:
        - cfg (:obj:`dict`):
            the config file, which is a multilevel dict, have large domain like
            evaluate, common, model, train etc, and each large domain
            has it's smaller domain.
        - wrapper_type (:obj:`str`): the type of wrapper returned, support ['time', 'cuda']

    Returns:
        - time_wrapper (:obj:`TimeWrapper`):
            return the corresponding timewrapper, reference nervex.utils.timehelper.TimeWrapperTime
            and nervex.utils.timehelper.get_cuda_time_wrapper
    """
    # Note: wrapper_type has higher priority
    if wrapper_type is not None:
        time_wrapper_type = wrapper_type
    elif cfg is not None:
        time_wrapper_type = cfg.common.time_wrapper_type
    else:
        raise RuntimeError('Either wrapper_type or cfg should be provided.')

    if time_wrapper_type == 'time' or (not torch.cuda.is_available()):
        return TimeWrapperTime
    elif time_wrapper_type == 'cuda':
        # lazy initialize to make code runnable locally
        return get_cuda_time_wrapper()
    else:
        raise KeyError('invalid time_wrapper_type: {}'.format(time_wrapper_type))


class EasyTimer:
    r"""
    Overview:
        A decent timer wrapper that can be used easily.

    Interface:
        __init__, __enter__, __exit__

    Example:
        >>> wait_timer = EasyTimer()
        >>> with wait_timer:
        >>>    func(...)
        >>> time_ = wait_timer.value  # in second
    """

    def __init__(self, cuda=True):
        r"""
        Overview:
            Init class EasyTimer

        Arguments:
            - cuda (:obj:`bool`): whether to build timer with cuda type
        """
        if torch.cuda.is_available() and cuda:
            time_wrapper_type = "cuda"
        else:
            time_wrapper_type = "time"
        self._timer = build_time_helper(wrapper_type=time_wrapper_type)
        self.value = 0.0

    def __enter__(self):
        r"""
        Overview:
            enter timer, start timing
        """
        self.value = 0.0
        self._timer.start_time()

    def __exit__(self, *args):
        r"""
        Overview:
            exit timer, stop timing
        """
        self.value = self._timer.end_time()


class TimeWrapper(object):
    r"""
    Overview:
        A abstract class method that defines TimeWrapper class

    Interface:
        wrapper, start_time, end_time
    """

    @classmethod
    def wrapper(cls, fn):
        r"""
        Overview:
            classmethod wrapper, wrap a function and automatically return its running time

        - fn (:obj:`function`): the function to be wrap and timed
        """

        def time_func(*args, **kwargs):
            cls.start_time()
            ret = fn(*args, **kwargs)
            t = cls.end_time()
            return ret, t

        return time_func

    @classmethod
    def start_time(cls):
        r"""
        Overview:
            abstract classmethod, start timing
        """
        raise NotImplementedError

    @classmethod
    def end_time(cls):
        r"""
        Overview:
            abstract classmethod, stop timing
        """
        raise NotImplementedError


class TimeWrapperTime(TimeWrapper):
    r"""
    Overview:
        A class method that inherit from TimeWrapper class

    Interface:
        start_time, end_time
    """

    # overwrite
    @classmethod
    def start_time(cls):
        r"""
        Overview:
            implement and overide the start_time method in TimeWrapper class
        """
        cls.start = time.time()

    # overwrite
    @classmethod
    def end_time(cls):
        r"""
        Overview:
            implement and overide the end_time method in TimeWrapper class

        Returns:
            - time(:obj:`float`): the time between start_time and end_time
        """
        cls.end = time.time()
        return cls.end - cls.start


def get_cuda_time_wrapper() -> Callable[[], 'TimeWrapper']:
    r"""
    Overview:
        Return the TimeWrapperCuda class

    Returns:
        - TimeWrapperCuda(:obj:`class`): see TimeWrapperCuda class
    """

    # TODO find a way to autodoc the class within method
    class TimeWrapperCuda(TimeWrapper):
        r"""
        Overview:
            A class method that inherit from TimeWrapper class

            Notes:
                must use torch.cuda.synchronize(), reference <https://blog.csdn.net/u013548568/article/details/81368019>

        Interface:
            start_time, end_time
        """
        # cls variable is initialized on loading this class
        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        # overwrite
        @classmethod
        def start_time(cls):
            r"""
            Overview:
                implement and overide the start_time method in TimeWrapper class
            """
            torch.cuda.synchronize()
            cls.start = cls.start_record.record()

        # overwrite
        @classmethod
        def end_time(cls):
            r"""
            Overview:
                implement and overide the end_time method in TimeWrapper class

            Returns:
                - time(:obj:`float`): the time between start_time and end_time
            """
            cls.end = cls.end_record.record()
            torch.cuda.synchronize()
            return cls.start_record.elapsed_time(cls.end_record) / 1000

    return TimeWrapperCuda
