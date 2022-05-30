from typing import TYPE_CHECKING, Union, Callable, Optional
import numpy as np
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def termination_checker(max_env_step: Optional[int] = None, max_train_iter: Optional[int] = None) -> Callable:
    if max_env_step is None:
        max_env_step = np.inf
    if max_train_iter is None:
        max_train_iter = np.inf

    def _check(ctx: Union["OnlineRLContext", "OfflineRLContext"]):
        # ">" is better than ">=" when taking logger result into consideration
        if ctx.env_step > max_env_step:
            task.finish = True
        if ctx.train_iter > max_train_iter:
            task.finish = True

    return _check
