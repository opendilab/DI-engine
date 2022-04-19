from typing import TYPE_CHECKING, Optional
from easydict import EasyDict
import os
import numpy as np

from ding.utils import save_file
from ding.policy import Policy
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context


class CkptSaver:

    def __init__(self, cfg: EasyDict, policy: Policy, train_freq: Optional[int] = None):
        self.policy = policy
        self.train_freq = train_freq
        self.prefix = '{}/ckpt'.format(cfg.exp_name)
        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)
        self.last_save_iter = 0
        self.max_eval_value = -np.inf

    def __call__(self, ctx: "Context") -> None:
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
