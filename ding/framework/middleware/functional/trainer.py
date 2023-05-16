from typing import TYPE_CHECKING, Callable, Union
from easydict import EasyDict
import treetensor.torch as ttorch
from ditk import logging
import numpy as np
from ding.policy import Policy
from ding.framework import task, OfflineRLContext, OnlineRLContext


def trainer(cfg: EasyDict, policy: Policy, log_freq: int = 100) -> Callable:
    """
    Overview:
        The middleware that executes a single training process.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be trained in step-by-step mode.
        - log_freq (:obj:`int`): The frequency (iteration) of showing log.
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
        if ctx.train_iter % log_freq == 0:
            if isinstance(train_output, list):
                train_output_loss = np.mean([item['total_loss'] for item in train_output])
            else:
                train_output_loss = train_output['total_loss']
            if isinstance(ctx, OnlineRLContext):
                logging.info(
                    'Training: Train Iter({})\tEnv Step({})\tLoss({:.3f})'.format(
                        ctx.train_iter, ctx.env_step, train_output_loss
                    )
                )
            elif isinstance(ctx, OfflineRLContext):
                logging.info('Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, train_output_loss))
            else:
                raise TypeError("not supported ctx type: {}".format(type(ctx)))
        ctx.train_iter += 1
        ctx.train_output = train_output

    return _train


def multistep_trainer(policy: Policy, log_freq: int = 100) -> Callable:
    """
    Overview:
        The middleware that executes training for a target num of steps.
    Arguments:
        - policy (:obj:`Policy`): The policy specialized for multi-step training.
        - log_freq (:obj:`int`): The frequency (iteration) of showing log.
    """
    last_log_iter = -1

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
        if hasattr(policy, "_device"):  # For ppof policy
            data = ctx.train_data.to(policy._device)
        elif hasattr(policy, "get_attribute"):  # For other policy
            data = ctx.train_data.to(policy.get_attribute("device"))
        else:
            assert AttributeError("Policy should have attribution '_device'.")
        train_output = policy.forward(data)
        nonlocal last_log_iter
        if ctx.train_iter - last_log_iter >= log_freq:
            loss = np.mean([o['total_loss'] for o in train_output])
            if isinstance(ctx, OfflineRLContext):
                logging.info('Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, loss))
            else:
                logging.info(
                    'Training: Train Iter({})\tEnv Step({})\tLoss({:.3f})'.format(ctx.train_iter, ctx.env_step, loss)
                )
            last_log_iter = ctx.train_iter
        ctx.train_iter += len(train_output)
        ctx.train_output = train_output

    return _train


# TODO reward model
