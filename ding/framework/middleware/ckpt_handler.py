from typing import TYPE_CHECKING, Optional, Union
from easydict import EasyDict
import os
import numpy as np

from ding.utils import save_file
from ding.policy import Policy
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


class CkptSaver:
    """
        Overview:
            The class to save checkpoint file.
    """

    def __init__(self, cfg: EasyDict, policy: Policy, train_freq: Optional[int] = None):
        """
        Overview:
            Initialize the CkptSaver.
        Arguments:
            - cfg (:obj:`EasyDict`): Config which should contain following keys: ['cfg.exp_name'].
            - policy (:obj:`Policy`): Policy.
            - train_freq (:obj:`int`): Number of training iterations every time saving checkpoint data.
        """
        self.policy = policy
        self.train_freq = train_freq
        self.prefix = '{}/ckpt'.format(cfg.exp_name)
        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)
        self.last_save_iter = 0
        self.max_eval_value = -np.inf

    def __call__(self, ctx: Union["OnlineRLContext", "OfflineRLContext"]) -> None:
        """
        Overview:
            The method to save checkpoint data. \
            The checkpoint data will be saved in a file in following 3 cases: \
                - Every time when train self.train_freq iterations; \
                - Every time when the eval reward is the best eval reward so far; \
                - When task.finish is True. \
        Input of ctx:
            - train_iter (:obj:`int`): Number of training iteration, i.e. the number of updating policy related network.
            - eval_value (:obj:`float`): The eval reward of current iteration.
        """
        # train enough iteration
        if self.train_freq and ctx.train_iter - self.last_save_iter >= self.train_freq:
            save_file(
                "{}/iteration_{}.pth.tar".format(self.prefix, ctx.train_iter), self.policy.learn_mode.state_dict()
            )
            self.last_save_iter = ctx.train_iter

        # best eval reward so far
        if ctx.eval_value > self.max_eval_value:
            save_file("{}/eval.pth.tar".format(self.prefix), self.policy.learn_mode.state_dict())
            self.max_eval_value = ctx.eval_value

        # finish
        if task.finish:
            save_file("{}/final.pth.tar".format(self.prefix), self.policy.learn_mode.state_dict())
