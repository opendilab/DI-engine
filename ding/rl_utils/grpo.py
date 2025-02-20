from typing import Tuple
from collections import namedtuple
import torch

grpo_policy_data = namedtuple('grpo_policy_data', ['logit_new', 'logit_old', 'logit_ref', 'action', 'adv', 'weight'])
MetricInfo = namedtuple('MetricInfo', ['mean_kl', 'mean_ratio', 'mean_clipped'])


def naive_method(logits, index):
    # Calculate log probabilities for each token
    log_prob_new = torch.log_softmax(logits, dim=-1)
    # Get log probabilities for selected actions
    index = index.unsqueeze(-1)  # [B, L, 1]
    per_token_logps = torch.gather(log_prob_new, -1, index).squeeze(-1)
    return per_token_logps


def efficient_method(logits, index):
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = torch.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def less_efficient_method(logits, action):
    dist = torch.distributions.categorical.Categorical(logits=logits)
    logp = dist.log_prob(action)
    return logp


def grpo_policy_error(
        data: namedtuple,
        logpro_cal=efficient_method,  # Method to calculate the log probabilities
        clip_ratio: float = 0.2,
        beta: float = 0.1  # Weight coefficient for KL divergence
) -> Tuple[namedtuple, namedtuple]:
    """
        Overview:
            Implementation of Generalized Reward-Conditioned Policy Optimization(	arXiv:2405.20304) .
        Arguments:
            - data (:obj:`namedtuple`): the grpo input data with fields shown in ``grpo_policy_data``.
            - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2.
            - beta (:obj:`float`): weight coefficient for KL divergence regularization, defaults to 0.1.
        Returns:
             - loss (:obj:`torch.FloatTensor`): the rloo policy loss, a differentiable 0-dim tensor.
            - grpo_info (:obj:`namedtuple`): the grpo optim information for monitoring, all of them are Python scalar.
        Shapes:
            - logit_new (:obj:`torch.FloatTensor`): :math:`(B, S, V)`, where B is batch size, S is sequence length,
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
    per_token_logps = logpro_cal(data.logit_new, data.action)
    per_token_ref_logps = logpro_cal(data.logit_ref, data.action)
    per_token_old_logps = logpro_cal(data.logit_old, data.action)

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
    metric_info = MetricInfo(
        mean_kl=((per_token_kl * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        mean_ratio=((ratio * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        mean_clipped=(ratio > (1 + clip_ratio)).float().mean().item() + (ratio <
                                                                         (1 - clip_ratio)).float().mean().item(),
    )

    return loss, metric_info
