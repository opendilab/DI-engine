from collections import namedtuple
import torch
from ding.hpc_rl import hpc_wrapper

gae_data = namedtuple('gae_data', ['value', 'next_value', 'reward', 'done', 'traj_flag'])
# gae_data_traj_flag = namedtuple('gae_data', ['value', 'next_value', 'reward', 'done', 'traj_flag'])


def shape_fn_gae(args, kwargs):
    r"""
    Overview:
        Return shape of gae for hpc
    Returns:
        shape: [T, B]
    """
    if len(args) <= 0:
        tmp = kwargs['data'].reward.shape
    else:
        tmp = args[0].reward.shape
    return tmp


@hpc_wrapper(
    shape_fn=shape_fn_gae, namedtuple_data=True, include_args=[0, 1, 2], include_kwargs=['data', 'gamma', 'lambda_']
)
def gae(data: namedtuple, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
    """
    Overview:
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
    Arguments:
        - data (:obj:`namedtuple`): gae input data with fields ['value', 'reward'], which contains some episodes or\
        trajectories data
        - gamma (:obj:`float`): the future discount factor, should be in [0, 1], defaults to 0.99.
        - lambda (:obj:`float`): the gae parameter lambda, should be in [0, 1], defaults to 0.97, when lambda -> 0,\
        it induces bias, but when lambda -> 1, it has high variance due to the sum of terms.
    Returns:
        - adv (:obj:`torch.FloatTensor`): the calculated advantage
    Shapes:
        - value (:obj:`torch.FloatTensor`): :math:`(T+1, B)`, where T is trajectory length and B is batch size
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - adv (:obj:`torch.FloatTensor`): :math:`(T, B)`

    .. note::
        value_{T+1} should be 0 if this trajectory reached a terminal state(done=True), otherwise we use value
        function, this operation is implemented in collector for packing trajectory.
    """
    value, next_value, reward, done, traj_flag = data
    if done is None:
        done = torch.zeros_like(reward, device=reward.device)
    if len(value.shape) == len(reward.shape) + 1:  # for some marl case: value(T, B, A), reward(T, B)
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
    delta = reward + (1 - done) * gamma * next_value - value
    factor = gamma * lambda_
    adv = torch.zeros_like(value, device=value.device)
    gae_item = torch.zeros_like(value[0])

    for t in reversed(range(reward.shape[0])):
        if traj_flag is None:
            gae_item = delta[t] + factor * gae_item * (1 - done[t])
        else:
            gae_item = delta[t] + factor * gae_item * (1 - traj_flag[t].float())
        adv[t] += gae_item
    return adv
