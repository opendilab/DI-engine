from collections import deque, defaultdict
from functools import wraps
from types import GeneratorType
from typing import Callable
import numpy as np
import time
from ditk import logging


class StepTimer:

    def __init__(self, print_per_step: int = 1, smooth_window: int = 10) -> None:
        self.print_per_step = print_per_step
        self.records = defaultdict(lambda: deque(maxlen=print_per_step * smooth_window))

    def __call__(self, fn: Callable) -> Callable:
        step_name = getattr(fn, "__name__", type(fn).__name__)

        @wraps(fn)
        def executor(ctx):
            start_time = time.time()
            time_cost = 0
            g = fn(ctx)
            if isinstance(g, GeneratorType):
                try:
                    next(g)
                except StopIteration:
                    pass
                time_cost = time.time() - start_time
                yield
                start_time = time.time()
                try:
                    next(g)
                except StopIteration:
                    pass
                time_cost += time.time() - start_time
            else:
                time_cost = time.time() - start_time
            self.records[step_name].append(time_cost * 1000)
            if ctx.total_step % self.print_per_step == 0:
                logging.info(
                    "[Step Timer] {}: Cost: {:.2f}ms, Mean: {:.2f}ms".format(
                        step_name, time_cost * 1000, np.mean(self.records[step_name])
                    )
                )

        return executor
