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
            The class used to save checkpoint data.
    """

    def __new__(cls, *args, **kwargs):
        if task.router.is_active and not (task.has_role(task.role.LEARNER) or task.has_role(task.role.EVALUATOR)):
            return task.void()
        return super(CkptSaver, cls).__new__(cls)

    def __init__(self, policy: Policy, save_dir: str, train_freq: Optional[int] = None, save_finish: bool = True):
        """
        Overview:
            Initialize the `CkptSaver`.
        Arguments:
            - policy (:obj:`Policy`): Policy used to save the checkpoint.
            - save_dir (:obj:`str`): The directory path to save ckpt.
            - train_freq (:obj:`int`): Number of training iterations between each saving checkpoint data.
            - save_finish (:obj:`bool`): Whether save final ckpt when ``task.finish = True``.
        """
        self.policy = policy
        self.train_freq = train_freq
        if str(os.path.basename(os.path.normpath(save_dir))) != "ckpt":
            self.prefix = '{}/ckpt'.format(os.path.normpath(save_dir))
        else:
            self.prefix = '{}/'.format(os.path.normpath(save_dir))
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)
        self.last_save_iter = 0
        self.max_eval_value = -np.inf
        self.save_finish = save_finish

    def __call__(self, ctx: Union["OnlineRLContext", "OfflineRLContext"]) -> None:
        """
        Overview:
            The method used to save checkpoint data. \
            The checkpoint data will be saved in a file in following 3 cases: \
                - When a multiple of `self.train_freq` iterations have elapsed since the beginning of training; \
                - When the evaluation episode return is the best so far; \
                - When `task.finish` is True.
        Input of ctx:
            - train_iter (:obj:`int`): Number of training iteration, i.e. the number of updating policy related network.
            - eval_value (:obj:`float`): The episode return of current iteration.
        """
        # train enough iteration
        if self.train_freq:
            if ctx.train_iter == 0 or ctx.train_iter - self.last_save_iter >= self.train_freq:
                save_file(
                    "{}/iteration_{}.pth.tar".format(self.prefix, ctx.train_iter), self.policy.learn_mode.state_dict()
                )
                self.last_save_iter = ctx.train_iter

        # best episode return so far
        if ctx.eval_value is not None and ctx.eval_value > self.max_eval_value:
            save_file("{}/eval.pth.tar".format(self.prefix), self.policy.learn_mode.state_dict())
            self.max_eval_value = ctx.eval_value

        # finish
        if task.finish and self.save_finish:
            save_file("{}/final.pth.tar".format(self.prefix), self.policy.learn_mode.state_dict())
