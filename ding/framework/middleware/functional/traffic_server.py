from typing import TYPE_CHECKING, Callable, Union, List
import time
import logging
from tabulate import tabulate
import atexit
from easydict import EasyDict
from ding.framework import task
from ding.framework.parallel import Parallel
from ding.utils import traffic
import pandas as pd
if TYPE_CHECKING:
    from ding.framework import Context


def traffic_server(
        cfg: EasyDict, process_dict: dict = None, online_analyse: bool = False, execution_rate: int = 1
) -> Callable:
    """
    Overview:
        Middleware for data analyser server as a master node that is both effective in local or remote mode. \
    Arguments:
        - cfg (:obj:`EasyDict`): Task configuration dictionary.
        - process_dict (:obj:`dict`): Dictionary of process functions for data to be processed.
        - online_analyse (:obj:`bool`): Whether to enable online analysis. 
        - execution_rate (:obj:`int`): Adjust the rate of execution.
    Returns:
        - _traffic_server_main (:obj:`Callable`): The main function.
    """

    assert execution_rate >= 1

    file_path = "./" + str(cfg.exp_name) + "/traffic_server/log.txt"

    traffic.set_config(file_path=file_path, online=online_analyse, router=Parallel())

    atexit.register(traffic.close)

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

        if online_analyse and process_dict:
            #if traffic.df.index.size > 10 and traffic.df.groupby("__label").ngroups >= 2 and ctx.traffic_step % execution_rate == 0:
            if traffic.df.index.size > 10 and ctx.traffic_step % execution_rate == 0:
                df_gb = traffic.df.groupby('__label')
                L = df_gb.agg(process_dict)
                logging.info(
                    tabulate(
                        L.stack().stack().reset_index().rename(columns={
                            "level_0": "",
                            "level_1": "",
                            "level_2": ""
                        }),
                        #headers="keys",
                        tablefmt='fancy_grid',
                        showindex=False
                    )
                )
        return

    return _traffic_server_main
