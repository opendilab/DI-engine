from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import logging
import numpy as np
from ding.policy import Policy
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context


def trainer(cfg: EasyDict, policy: Policy) -> Callable:

    def _train(ctx: "Context"):

        if ctx.train_data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            logging.info(
                'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, train_output['total_loss'])
            )
        ctx.train_iter += 1
        ctx.train_output = train_output

    return _train


def multistep_trainer(cfg: EasyDict, policy: Policy) -> Callable:

    def _train(ctx: "Context"):

        if ctx.train_data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            loss = np.mean([o['total_loss'] for o in train_output])
            logging.info('Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, loss))
        ctx.train_iter += len(train_output)
        ctx.train_output = train_output

    return _train


# TODO reward model
