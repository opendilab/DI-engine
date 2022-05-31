from typing import TYPE_CHECKING, Callable, Union
from easydict import EasyDict
from ditk import logging
import numpy as np
from ding.policy import Policy
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def trainer(cfg: EasyDict, policy: Policy) -> Callable:
    """
    Overview:
        The middleware that executes a single training process.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be trained in step-by-step mode.
    """

    def _train(ctx: Union["OnlineRLContext", "OfflineRLContext"]):
        """
        Input of ctx:
            - train_data (:obj:`Dict`): The data used to update the network. It will train only if \
                the data is not empty.
            - train_iter: (:obj:`int`): The training iteration count. The log will be printed once \
                it reachs certain values.
        Output of ctx:
            - train_output (:obj:`Dict`): The training output in the Dict format, including loss info.
        """

        if ctx.train_data is None:
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            logging.info(
                'Training: Train Iter({})\tEnv Step({})\tLoss({:.3f})'.format(
                    ctx.train_iter, ctx.env_step, train_output['total_loss']
                )
            )
        ctx.train_iter += 1
        ctx.train_output = train_output

    return _train


def multistep_trainer(cfg: EasyDict, policy: Policy) -> Callable:
    """
    Overview:
        The middleware that executes training for a target num of steps.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy specialized for multi-step training.
    """

    def _train(ctx: Union["OnlineRLContext", "OfflineRLContext"]):
        """
        Input of ctx:
            - train_data: The data used to update the network.
                It will train only if the data is not empty.
            - train_iter: (:obj:`int`): The training iteration count.
                The log will be printed if it reachs certain values.
        Output of ctx:
            - train_output (:obj:`List[Dict]`): The training output listed by steps.
        """

        if ctx.train_data is None:  # no enough data from data fetcher
            return
        train_output = policy.forward(ctx.train_data)
        if ctx.train_iter % cfg.policy.learn.learner.hook.log_show_after_iter == 0:
            loss = np.mean([o['total_loss'] for o in train_output])
            logging.info(
                'Training: Train Iter({})\tEnv Step({})\tLoss({:.3f})'.format(ctx.train_iter, ctx.env_step, loss)
            )
        ctx.train_iter += len(train_output)
        ctx.train_output = train_output

    return _train


# TODO reward model
