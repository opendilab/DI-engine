from typing import TYPE_CHECKING, Callable, Union, List
import time
import logging
from tabulate import tabulate
import atexit
from easydict import EasyDict
from ding.framework import task
from ding.utils import traffic
import pandas as pd
import numpy as np
if TYPE_CHECKING:
    from ding.framework import Context


def traffic_server(execution_period: int = 1) -> Callable:
    """
    Overview:
        Middleware for data analyser server as a master node that is both effective in local or remote mode. \
    Arguments:
        - execution_period (:obj:`int`): Adjust the rate of execution.
    Returns:
        - _traffic_server_main (:obj:`Callable`): The main function.
    """

    assert execution_period >= 1

    def _traffic_server_main(ctx: "Context") -> None:
        """
        Overview:
            Make online analysis if needed. \
                Listen and record data and save it offline.
        Arguments:
            - ctx (:obj:`Context`): Context of task object.
        """

        if "traffic_step" in ctx:
            ctx.traffic_step += 1
        else:
            ctx.traffic_step = 0

        if traffic.df.index.size > 10 and ctx.traffic_step % execution_period == 0:
            L = traffic.df.drop(["__label", "__time"],
                                axis=1).replace('', np.nan).ffill().iloc[-1].to_frame(name="values").T
            L = L[~L.applymap(pd.api.types.is_list_like)].dropna(axis=1, how="all")
            len = L.shape[1]
            for col in range(0, len, 5):
                col_end = min(col + 5, len)
                logging.info(tabulate(L.iloc[:, col:col_end], headers="keys", tablefmt='fancy_grid', showindex=False))

        return

    return _traffic_server_main
