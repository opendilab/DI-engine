from typing import Tuple
from collections import namedtuple
import torch
from .log_prob_utils import efficient_method, naive_method, less_efficient_method, LogProbFunction

rloo_policy_data = namedtuple('rloo_policy_data', ['logit_new', 'logit_old', 'action', 'reward', 'weight'])
rloo_info = namedtuple('rloo_info', ['approx_kl', 'clipfrac'])


def rloo_policy_error(
        data: namedtuple,
        log_prob_fn: LogProbFunction = efficient_method,  # Method to calculate the log probabilities
        clip_ratio: float = 0.2,
) -> Tuple[namedtuple, namedtuple]:
    """
    Overview:
        REINFORCE Leave-One-Out(arXiv:2402.14740)
    Arguments:
        - data (:obj:`namedtuple`): the rloo input data with fields shown in ``rloo_policy_data``.
        - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2.
        - log_prob_fn (:obj:`LogProbFunction`): The method to calculate the log probabilities, \
             defaults to `efficient_method`.
    Returns:
        - loss (:obj:`torch.FloatTensor`): the rloo policy loss, a differentiable 0-dim tensor.
        - rloo_info (:obj:`namedtuple`): the rloo optim information for monitoring, all of them are Python scalar.
    Shapes:
        - logit_new (:obj:`torch.FloatTensor`): :math:`(B, S, V)`, where B is batch size, S is sequence length,\
              and V is vocabulary size.
        - logit_old (:obj:`torch.FloatTensor`): :math:`(B, S, V)`.
        - action (:obj:`torch.LongTensor`): :math:`(B, S)`.
        - reward (:obj:`torch.FloatTensor`): :math:`(K, B)`, where K is the number of samples per prompt.
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, S)`.
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor.
        - mean_ratio (:obj:`float`): mean probability ratio.
        - mean_clipped (:obj:`float`): proportion of clipped probability ratios.
        - mean_advantage (:obj:`float`): mean advantage value.
    """

    # Calculate advantage of each action
    rloo_k = data.reward.size(0)
    baseline = (data.reward.sum(0) - data.reward) / (rloo_k - 1)
    adv = data.reward - baseline
    adv = adv.flatten()

    # Get log probabilities for selected actions
    per_token_logps = log_prob_fn(data.logit_new, data.action)
    per_token_old_logps = log_prob_fn(data.logit_old, data.action)

    # Calculate policy ratio
    ratio = torch.exp(per_token_logps - per_token_old_logps)
    ratio_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Calculate loss for each token
    advantages = adv.unsqueeze(1)  # [B, 1]
    per_token_loss_unclipped = ratio * advantages
    per_token_loss_clipped = ratio_clipped * advantages
    per_token_loss = -torch.min(per_token_loss_unclipped, per_token_loss_clipped)

    # Calculate average loss using weight mask
    weight = data.weight if data.weight is not None else (torch.ones_like(per_token_loss))
    loss = ((per_token_loss * weight).sum(dim=1) / weight.sum(dim=1)).mean()

    # Calculate additional metrics
    with torch.no_grad():
        approx_kl = (per_token_old_logps - per_token_logps).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()

    return loss, rloo_info(approx_kl=approx_kl, clipfrac=clipfrac)
