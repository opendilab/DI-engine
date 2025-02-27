from typing import Tuple
from collections import namedtuple
import torch
from .log_prob_utils import efficient_method, naive_method, less_efficient_method, LogProbFunction

grpo_policy_data = namedtuple('grpo_policy_data', ['logit_new', 'logit_old', 'logit_ref', 'action', 'adv', 'weight'])
grpo_info = namedtuple('grpo_info', ['approx_kl', 'clipfrac'])


def grpo_policy_error(
        data: namedtuple,
        log_prob_fn: LogProbFunction = efficient_method,  # Method to calculate the log probabilities
        clip_ratio: float = 0.2,
        beta: float = 0.1  # Weight coefficient for KL divergence
) -> Tuple[namedtuple, namedtuple]:
    """
        Overview:
            Implementation of Generalized Reward-Conditioned Policy Optimization(	arxiv:2402.03300) .
        Arguments:
            - data (:obj:`namedtuple`): the grpo input data with fields shown in ``grpo_policy_data``.
            - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2.
            - beta (:obj:`float`): weight coefficient for KL divergence regularization, defaults to 0.1.
            - log_prob_fn (:obj:`LogProbFunction`): The method to calculate the log probabilities, \
                  defaults to `efficient_method`.
        Returns:
            - loss (:obj:`torch.FloatTensor`): the rloo policy loss, a differentiable 0-dim tensor.
            - grpo_info (:obj:`namedtuple`): the grpo optim information for monitoring, all of them are Python scalar.
        Shapes:
            - logit_new (:obj:`torch.FloatTensor`): :math:`(B, S, V)`, where B is batch size, S is sequence length, \
                   and V is vocabulary size.
            - logit_old (:obj:`torch.FloatTensor`): :math:`(B, S, V)`.
            - logit_ref (:obj:`torch.FloatTensor`): :math:`(B, S, V)`.
            - action (:obj:`torch.LongTensor`): :math:`(B, S)`.
            - adv (:obj:`torch.FloatTensor`): :math:`(B, )`.
            - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, S)`.
            - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor.
            - mean_kl (:obj:`float`): mean KL divergence between current and reference policy.
            - mean_ratio (:obj:`float`): mean probability ratio.
            - mean_clipped (:obj:`float`): proportion of clipped probability ratios.
        """

    # Calculate log probabilities for selected token
    per_token_logps = log_prob_fn(data.logit_new, data.action)
    per_token_ref_logps = log_prob_fn(data.logit_ref, data.action)
    per_token_old_logps = log_prob_fn(data.logit_old, data.action)

    # Calculate KL divergence: exp(q-p) - (q-p) - 1,
    # where p is current policy and q is reference policy
    per_token_kl = (torch.exp(per_token_ref_logps - per_token_logps) - (per_token_ref_logps - per_token_logps) - 1)

    # Calculate policy ratio
    ratio = torch.exp(per_token_logps - per_token_old_logps)
    ratio_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # Calculate loss for each token
    advantages = data.adv.unsqueeze(1)  # [B, 1]
    per_token_loss_unclipped = ratio * advantages
    per_token_loss_clipped = ratio_clipped * advantages
    per_token_loss = -torch.min(per_token_loss_unclipped, per_token_loss_clipped)

    # Add KL divergence regularization term
    per_token_loss = per_token_loss + beta * per_token_kl

    # Calculate average loss using weight mask
    weight = data.weight if data.weight is not None \
        else torch.ones_like(per_token_loss)
    loss = ((per_token_loss * weight).sum(dim=1) / weight.sum(dim=1)).mean()

    # Calculate additional metrics
    with torch.no_grad():
        approx_kl = (per_token_old_logps - per_token_logps).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()

    return loss, grpo_info(approx_kl=approx_kl, clipfrac=clipfrac)
