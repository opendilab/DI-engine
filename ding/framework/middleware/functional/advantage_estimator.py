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
from ding.torch_utils import to_device

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
    obs_shape = cfg['policy']['model']['obs_shape']
    obs_shape = torch.Size(torch.tensor(obs_shape)) if isinstance(obs_shape, list) \
        else torch.Size(torch.tensor(obs_shape).unsqueeze(0))
    action_shape = cfg['policy']['model']['action_shape']
    action_shape = torch.Size(torch.tensor(action_shape)) if isinstance(action_shape, list) \
        else torch.Size(torch.tensor(action_shape).unsqueeze(0))

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
        cuda = cfg.policy.cuda and torch.cuda.is_available()

        # action shape (B,) for discete action, (B, D,) for continuous action
        # reward shape (B,) done shape (B,) value shape (B,)
        data = ttorch_collate(ctx.trajectories, cat_1dim=True)
        if data['action'].dtype in [torch.float16,torch.float32,torch.double] \
            and data['action'].dim() == 1 :
            # action shape
            data['action'] = data['action'].unsqueeze(-1)

        with torch.no_grad():
            if cuda:
                data = data.cuda()
            value = model.forward(data.obs.to(dtype=ttorch.float32), mode='compute_critic')['value']
            next_value = model.forward(data.next_obs.to(dtype=ttorch.float32), mode='compute_critic')['value']
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
            data = data.cpu()
            data = ttorch.split(data, 1)
            # To ensure the shape of obs is same as config
            if data[0]['obs'].shape == obs_shape:
                pass
            elif data[0]['obs'].shape[0] == 1 and data[0]['obs'].shape[1:] == obs_shape:
                for d in data:
                    d['obs'] = d['obs'].squeeze(0)
                    d['next_obs'] = d['next_obs'].squeeze(0)
                if 'logit' in data[0]:
                    for d in data:
                        d['logit'] = d['logit'].squeeze(0)
                if 'log_prob' in data[0]:
                    for d in data:
                        d['log_prob'] = d['log_prob'].squeeze(0)
            else:
                raise RuntimeError("The shape of obs is {}, which is not same as config.".format(data[0]['obs'].shape))

            if data[0]['action'].dtype in [torch.float16,torch.float32,torch.double] \
                    and data[0]['action'].dim() == 2:
                for d in data:
                    d['action'] = d['action'].squeeze(0)
            for d in data:
                buffer_.push(d)
        ctx.trajectories = None

    return _gae


def ppof_adv_estimator(policy: Policy) -> Callable:

    def _estimator(ctx: "OnlineRLContext"):
        data = ttorch_collate(ctx.trajectories, cat_1dim=True)
        if data['action'].dtype in [torch.float16,torch.float32,torch.double] \
            and data['action'].dim() == 1 :
            data['action'] = data['action'].unsqueeze(-1)
        traj_flag = data.done.clone()
        traj_flag[ctx.trajectory_end_idx] = True
        data.traj_flag = traj_flag
        ctx.train_data = data

    return _estimator


def pg_estimator(policy: Policy) -> Callable:

    def _estimator(ctx: "OnlineRLContext"):
        train_data = []
        for episode in ctx.episodes:
            data = ttorch_collate(episode, cat_1dim=True)
            if data['action'].dtype in [torch.float16,torch.float32,torch.double] \
                and data['action'].dim() == 1 :
                data['action'] = data['action'].unsqueeze(-1)
            data = policy.get_train_sample(data)
            train_data.append(data)
        ctx.train_data = ttorch.cat(train_data, dim=0)

    return _estimator
