import numpy as np
from collections import deque
from ditk import logging
from time import time

from ding.framework import task
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework.context import Context


def epoch_timer(print_per: int = 1, smooth_window: int = 10):
    """
    Overview:
        Print time cost of each epoch.
    Arguments:
        - print_per (:obj:`int`): Print each N epoch.
        - smooth_window (:obj:`int`): The window size to smooth the mean.
    """
    records = deque(maxlen=print_per * smooth_window)

    def _epoch_timer(ctx: "Context"):
        start = time()
        yield
        time_cost = time() - start
        records.append(time_cost)
        if ctx.total_step % print_per == 0:
            logging.info(
                "[Epoch Timer][Node:{:>2}]: Cost: {:.2f}ms, Mean: {:.2f}ms".format(
                    task.router.node_id or 0, time_cost * 1000,
                    np.mean(records) * 1000
                )
            )

    return _epoch_timer
