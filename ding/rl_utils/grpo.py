from typing import Tuple
from collections import namedtuple
import torch

grpo_policy_data = namedtuple('grpo_policy_data', ['logit_new', 'logit_old', 'logit_ref', 'action', 'adv', 'weight'])


def grpo_policy_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        beta: float = 0.1,  # Weight coefficient for KL divergence
) -> Tuple[namedtuple, namedtuple]:
    """Calculate the policy loss for GRPO
    Args:
        data (grpo_policy_data): Data containing the following fields:
            - logit_new: Current policy logits [B, L, V]
            - logit_old: Old policy logits [B, L, V]
            - logit_ref: Reference policy logits [B, L, V]
            - action: Actions taken [B, L]
            - adv: Advantage values [B]
            - weight: Attention mask [B, L]
        clip_ratio (float): PPO clipping ratio, default 0.2
        beta (float): Weight coefficient for KL divergence, default 0.1

    Returns:
        Tuple[namedtuple, namedtuple]:
            - First namedtuple contains policy_loss
            - Second namedtuple contains additional metrics
    """

    # Calculate log probabilities for each token
    log_prob_new = torch.log_softmax(data.logit_new, dim=-1)
    log_prob_old = torch.log_softmax(data.logit_old, dim=-1)
    log_prob_ref = torch.log_softmax(data.logit_ref, dim=-1)

    # Get log probabilities for selected actions
    action = data.action.unsqueeze(-1)  # [B, L, 1]
    per_token_logps = torch.gather(log_prob_new, -1, action).squeeze(-1)
    per_token_old_logps = torch.gather(log_prob_old, -1, action).squeeze(-1)
    per_token_ref_logps = torch.gather(log_prob_ref, -1, action).squeeze(-1)

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
    metrics = {
        'mean_kl': ((per_token_kl * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        'mean_ratio': ((ratio * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        'mean_clipped': (
            (ratio > (1 + clip_ratio)).float().mean().item() + (ratio < (1 - clip_ratio)).float().mean().item()
        ),
    }

    # Create return namedtuples
    loss_info = namedtuple('LossInfo', ['policy_loss'])(policy_loss=loss)
    metric_info = namedtuple('MetricInfo', list(metrics.keys()))(**metrics)

    return loss_info, metric_info
