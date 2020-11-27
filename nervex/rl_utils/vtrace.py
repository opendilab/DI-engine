import torch
import torch.nn.functional as F
from collections import namedtuple
from .isw import compute_importance_weights


def vtrace_nstep_return(clipped_rhos, clipped_cs, reward, bootstrap_values, gamma=0.99, lambda_=0.95):
    deltas = clipped_rhos * (reward + gamma * bootstrap_values[1:] - bootstrap_values[:-1])
    factor = gamma * lambda_
    result = bootstrap_values[:-1].clone()
    vtrace_item = 0.
    for t in reversed(range(reward.size()[0] - 1)):
        vtrace_item = deltas[t] + factor * clipped_cs[t] * vtrace_item
        result[t] += vtrace_item
    return result


def vtrace_advantage(clipped_pg_rhos, reward, return_, bootstrap_values, gamma):
    return clipped_pg_rhos * (reward + gamma * return_ - bootstrap_values)


vtrace_data = namedtuple('vtrace_data', ['target_output', 'behaviour_output', 'action', 'value', 'reward', 'weight'])
vtrace_loss = namedtuple('vtrace_loss', ['policy_loss', 'value_loss', 'entropy_loss'])


def vtrace_error(
    data: namedtuple,
    gamma: float = 0.99,
    lambda_: float = 1.9,
    rho_clip_ratio: float = 1.0,
    c_clip_ratio: float = 1.0,
    rho_pg_clip_ratio: float = 1.0
):
    """
    Overview:
        Implementation of vtrace(IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner\
        Architectures), (arXiv:1802.01561)
    Arguments:
        - data (:obj:`namedtuple`): input data with fieids shown in ``vtrace_data``
            - target_output (:obj:`torch.Tensor`): the output taking the action by the current policy network,\
                usually this output is network output logit
            - behaviour_output (:obj:`torch.Tensor`): the output taking the action by the behaviour policy network,\
                usually this output is network output logit, which is used to produce the trajectory(actor)
            - action (:obj:`torch.Tensor`): the chosen action(index for the discrete action space) in trajectory,\
                i.e.: behaviour_action
        - gamma: (:obj:`float`): the future discount factor, defaults to 0.95
        - lambda: (:obj:`float`): mix factor between 1-step (lambda_=0) and n-step, defaults to 1.0
        - rho_clip_ratio (:obj:`float`): the clipping threshold for importance weights (rho) when calculating\
            the baseline targets (vs)
        - c_clip_ratio (:obj:`float`): the clipping threshold for importance weights (c) when calculating\
            the baseline targets (vs)
        - rho_pg_clip_ratio (:obj:`float`): the clipping threshold for importance weights (rho) when calculating\
            the policy gradient advantage
    Returns:
        - trace_loss (:obj:`namedtuple`): the vtrace loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - target_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where T is timestep, B is batch size and\
            N is action dim
        - behaviour_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(T, B)`
        - value (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
        - reward (:obj:`torch.LongTensor`): :math:`(T, B)`
        - weight (:obj:`torch.LongTensor`): :math:`(T, B)`
    """
    target_output, behaviour_output, action, value, reward, weight = data
    with torch.no_grad():
        IS = compute_importance_weights(target_output, behaviour_output, action)
        rhos = torch.clamp(IS, max=rho_clip_ratio)
        cs = torch.clamp(IS, max=c_clip_ratio)
        return_ = vtrace_nstep_return(rhos, cs, reward, value, gamma, lambda_)
        pg_rhos = torch.clamp(IS, max=rho_pg_clip_ratio)
        adv = vtrace_advantage(pg_rhos, reward, return_, value[:-1], gamma)

    if weight is None:
        weight = torch.ones_like(reward)
    dist_target = torch.distributions.Categorical(logits=target_output)
    pg_loss = -(dist_target.log_prob(action) * adv * weight).mean()
    value_loss = (F.mse_loss(value[:-1], return_, reduction='none') * weight).mean()
    entropy_loss = (dist_target.entropy() * weight).mean()
    return vtrace_loss(pg_loss, value_loss, entropy_loss)
