from typing import Callable
import torch
from .time_helper_base import TimeWrapper


def get_cuda_time_wrapper() -> Callable[[], 'TimeWrapper']:
    r"""
    Overview:
        Return the ``TimeWrapperCuda`` class, this wrapper aims to ensure compatibility in no cuda device

    Returns:
        - TimeWrapperCuda(:obj:`class`): See ``TimeWrapperCuda`` class

    .. note::
        Must use ``torch.cuda.synchronize()``, reference: <https://blog.csdn.net/u013548568/article/details/81368019>

    """

    # TODO find a way to autodoc the class within method
    class TimeWrapperCuda(TimeWrapper):
        r"""
        Overview:
            A class method that inherit from ``TimeWrapper`` class

            Notes:
                Must use torch.cuda.synchronize(), reference: \
                <https://blog.csdn.net/u013548568/article/details/81368019>

        Interface:
            ``start_time``, ``end_time``
        """
        # cls variable is initialized on loading this class
        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        # overwrite
        @classmethod
        def start_time(cls):
            r"""
            Overview:
                Implement and overide the ``start_time`` method in ``TimeWrapper`` class
            """
            torch.cuda.synchronize()
            cls.start = cls.start_record.record()

        # overwrite
        @classmethod
        def end_time(cls):
            r"""
            Overview:
                Implement and overide the end_time method in ``TimeWrapper`` class
            Returns:
                - time(:obj:`float`): The time between ``start_time`` and ``end_time``
            """
            cls.end = cls.end_record.record()
            torch.cuda.synchronize()
            return cls.start_record.elapsed_time(cls.end_record) / 1000

    return TimeWrapperCuda
