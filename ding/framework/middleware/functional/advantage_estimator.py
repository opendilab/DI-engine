from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import logging
import treetensor.torch as ttorch
from ding.policy import Policy
from ding.rl_utils import gae, gae_data
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


def collate(x):  # TODO ttorch.collate
    x = ttorch.stack(x)
    for k in x.keys():
        if len(x[k].shape) == 2 and x[k].shape[-1] == 1:
            x[k] = x[k].squeeze(-1)
    return x


def gae_estimator(cfg: EasyDict, policy: Policy) -> Callable:

    def _gae(ctx: "OnlineRLContext"):

        data = ctx.trajectories  # list
        data = collate(data)
        next_value = data.value[1:]
        data = data[:-1]

        traj_flag = ttorch.zeros(len(data.done)).to(data.done.device)
        traj_flag[ctx.trajectory_end_idx[:-1]] = 1
        data.traj_flag = traj_flag

        # done is bool type when acquired from env.step
        data_ = gae_data(data.value, next_value, data.reward, data.done.float(), traj_flag)
        data.adv = gae(data_, cfg.policy.collect.discount_factor, cfg.policy.collect.gae_lambda)
        ctx.train_data = data

    return _gae
