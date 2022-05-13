from typing import TYPE_CHECKING, Callable, Optional
from easydict import EasyDict
from ditk import logging
import torch
import treetensor.torch as ttorch
from ding.policy import Policy
from ding.data import Buffer
from ding.rl_utils import gae, gae_data
from ding.framework import task
from ding.utils.data import ttorch_collate

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


def gae_estimator(cfg: EasyDict, policy: Policy, buffer_: Optional[Buffer] = None) -> Callable:
    """
    Overview:
        Calculate value using observation of input data, then call function `gae` to get advantage. \
        The processed data will be pushed into `buffer_` if `buffer_` is not None, \
        otherwise it will be assigned to `ctx.train_data`.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys: \
            `cfg.policy.collect.discount_factor`, `cfg.policy.collect.gae_lambda`.
        - policy (:obj:`Policy`): Policy in `policy.collect_mode`, used to get model to calculate value.
        - buffer\_ (:obj:`Optional[Buffer]`): The `buffer_` to push the processed data in if `buffer_` is not None.
    """

    model = policy.get_attribute('model')

    def _gae(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[treetensor.torch.Tensor]`): The data to be processed.\
                Each element should contain the following keys: `obs`, `next_obs`, `reward`, `done`.
            - trajectory_end_idx: (:obj:`treetensor.torch.IntTensor`):
                The indices that define the end of trajectories, \
                which should be shorter than the length of `ctx.trajectories`.
        Output of ctx:
            - train_data (:obj:`List[treetensor.torch.Tensor]`): The processed data if `buffer_` is None.
        """
        data = ctx.trajectories  # list
        data = ttorch_collate(data)
        with torch.no_grad():
            value = model.forward(data.obs, mode='compute_critic')['value']
            next_value = model.forward(data.next_obs, mode='compute_critic')['value']
            data.value = value

        traj_flag = data.done.clone()
        traj_flag[ctx.trajectory_end_idx] = True
        data.traj_flag = traj_flag

        # done is bool type when acquired from env.step
        data_ = gae_data(data.value, next_value, data.reward, data.done.float(), traj_flag.float())
        data.adv = gae(data_, cfg.policy.collect.discount_factor, cfg.policy.collect.gae_lambda)
        if buffer_ is None:
            ctx.train_data = data
        else:
            data = ttorch.split(data, 1)
            for d in data:
                buffer_.push(d)
        ctx.trajectories = None

    return _gae
