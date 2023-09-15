from typing import TYPE_CHECKING, Union, Callable, Optional
from ditk import logging
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
        assert hasattr(ctx, "env_step") or hasattr(ctx, "train_iter"), "Context must have env_step or train_iter"
        if hasattr(ctx, "env_step") and ctx.env_step > max_env_step:
            task.finish = True
            logging.info('Exceeded maximum number of env_step({}), program is terminated'.format(ctx.env_step))
        elif hasattr(ctx, "train_iter") and ctx.train_iter > max_train_iter:
            task.finish = True
            logging.info('Exceeded maximum number of train_iter({}), program is terminated'.format(ctx.train_iter))

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
                logging.info('Exceeded maximum number of env_step({}), program is terminated'.format(ctx.env_step))
            elif ctx.train_iter > max_train_iter:
                finish = torch.ones(1).long().cuda()
                logging.info('Exceeded maximum number of train_iter({}), program is terminated'.format(ctx.train_iter))
            else:
                finish = torch.LongTensor([task.finish]).cuda()
        else:
            finish = torch.zeros(1).long().cuda()
        # broadcast finish result to other DDP workers
        broadcast(finish, 0)
        task.finish = finish.cpu().bool().item()

    return _check
