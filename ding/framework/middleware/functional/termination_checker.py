from typing import TYPE_CHECKING, Union, Callable, Optional
import numpy as np
import torch
from ding.utils import broadcast
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


def ddp_termination_checker(max_env_step=None, max_train_iter=None, rank=0):
    if rank == 0:
        if max_env_step is None:
            max_env_step = np.inf
        if max_train_iter is None:
            max_train_iter = np.inf

    def _check(ctx):
        if rank == 0:
            if ctx.env_step > max_env_step:
                finish = torch.ones(1).long().cuda()
            elif ctx.train_iter > max_train_iter:
                finish = torch.ones(1).long().cuda()
            else:
                finish = torch.LongTensor([task.finish]).cuda()
        else:
            finish = torch.zeros(1).long().cuda()
        broadcast(finish, 0)
        task.finish = finish.cpu().bool().item()

    return _check
