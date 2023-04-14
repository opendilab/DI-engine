import torch
import functools
import time
from concurrent import futures
from ditk import logging
from typing import List, Optional, Tuple, Dict, Any
from ding.utils import EasyTimer, byte_beauty_print

# Data size for some tests
UNIT_1_B = 1
UNIT_1_KB = 1024 * UNIT_1_B
UNIT_1_MB = 1024 * UNIT_1_KB
UNIT_1_GB = 1024 * UNIT_1_MB
TENSOR_SIZE_LIST = [
    8 * UNIT_1_B, 32 * UNIT_1_B, 64 * UNIT_1_B, UNIT_1_KB, 4 * UNIT_1_KB, 64 * UNIT_1_KB, 1 * UNIT_1_MB, 4 * UNIT_1_MB,
    64 * UNIT_1_MB, 512 * UNIT_1_MB, 1 * UNIT_1_GB, 2 * UNIT_1_GB, 4 * UNIT_1_GB
]

# TODO: Add perf switch to avoid performance loss to critical paths during non-test time.
DO_PERF = False

# Convert from torch.dtype to bytes
TYPE_MAP = {torch.float32: 4, torch.float64: 8, torch.int32: 4, torch.int64: 8, torch.uint8: 1}

# A list of time units and names.
TIME_UNIT = [1, 1000, 1000]
TIME_NAME = ["s", "ms", "us"]

# The global function timing result is stored in OUTPUT_DICT.
OUTPUT_DICT = dict()


def _store_timer_result(func_name: str, avg_tt: float):
    if func_name not in OUTPUT_DICT.keys():
        OUTPUT_DICT[func_name] = str(round(avg_tt, 4)) + ","
    else:
        OUTPUT_DICT[func_name] = OUTPUT_DICT[func_name] + str(round(avg_tt, 4)) + ","


def print_timer_result_csv():
    """
        Overview:
            Output the average execution time of all functions durning this
            experiment in csv format.
    """
    for key, value in OUTPUT_DICT.items():
        print("{},{}".format(key, value))


def time_perf_once(unit: int, cuda: bool = False):
    """
    Overview:
        Decorator function to measure the time of a function execution.
    Arguments:
        - unit ([int]): 0 for s timer, 1 for ms timer, 2 for us timer.
        - cuda (bool, optional): Whether CUDA operation occurred within the timing range.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            timer = EasyTimer(cuda=cuda)
            with timer:
                func(*args, **kw)
            tt = timer.value * TIME_UNIT[unit]
            logging.info("func:\"{}\" use {:.4f} {},".format(func.__name__, tt, TIME_NAME[unit]))

            _store_timer_result(func.__name__, tt)

        return wrapper

    return decorator


def time_perf_avg(unit: int, count: int, skip_iter: int = 0, cuda: bool = False):
    """
    Overview:
        A decorator that averages the execution time of a function.
    Arguments:
        - unit (int): 0 for s timer, 1 for ms timer, 2 for us timer
        - time_list (List): User-supplied list for staging execution times.
        - count (int): Loop count.
        - skip_iter (int, optional): Skip the first n iter times.
        - cuda (bool, optional): Whether CUDA operation occurred within the timing range.
    """
    time_list = []

    if skip_iter >= count:
        logging.error("skip_iter:[{}] must >= count:[{}]".format(skip_iter, count))
        return None

    def decorator(func):

        @functools.wraps(func)
        def wrapper(idx, *args, **kw):
            timer = EasyTimer(cuda=cuda)
            with timer:
                func(*args, **kw)

            if idx < skip_iter:
                return

            time_list.append(timer.value * TIME_UNIT[unit])
            if idx == count - 1:
                avg_tt = sum(time_list) / len(time_list)
                logging.info(
                    "\"{}\": repeat[{}], avg_time[{:.4f}]{},".format(
                        func.__name__, len(time_list), avg_tt, TIME_NAME[unit]
                    )
                )

                _store_timer_result(func.__name__, avg_tt)
                time_list.clear()

        return wrapper

    return decorator


def dtype_2_byte(dtype: torch.dtype) -> int:
    return TYPE_MAP[dtype]


def tensor_size_beauty_print(length: int, dtype: torch.dtype) -> tuple:
    return byte_beauty_print(length * dtype_2_byte(dtype))
