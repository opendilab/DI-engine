from typing import Tuple
from collections import namedtuple
import torch

rloo_policy_data = namedtuple('rloo_policy_data', ['logit_new', 'logit_old', 'action', 'reward', 'weight'])


def rloo_policy_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
) -> Tuple[namedtuple, namedtuple]:
    """Calculate the policy loss for RLOO

       Args:
           data (rloo_policy_data): Data containing the following fields:
               - logit_new: Current policy logits [B, L, V]
               - logit_old: Old policy logits [B, L, V]
               - action: Actions taken [B, L]
               - reward: Advantage values [B]
               - weight: Attention mask [B, L]
           clip_ratio (float): PPO clipping ratio, default 0.2

       Returns:
           Tuple[namedtuple, namedtuple]:
               - First namedtuple contains policy_loss
               - Second namedtuple contains additional metrics
       """
    # Calculate advantage of each action
    rloo_k = data.reward.size(0)
    baseline = (data.reward.sum(0) - data.reward) / (rloo_k - 1)
    adv = data.reward - baseline
    adv = adv.flatten()

    # Calculate log probabilities for each token
    log_prob_new = torch.log_softmax(data.logit_new, dim=-1)
    log_prob_old = torch.log_softmax(data.logit_old, dim=-1)

    # Get log probabilities for selected actions
    action = data.action.unsqueeze(-1)  # [B, L, 1]
    per_token_logps = torch.gather(log_prob_new, -1, action).squeeze(-1)
    per_token_old_logps = torch.gather(log_prob_old, -1, action).squeeze(-1)

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
    metrics = {
        'mean_ratio': ((ratio * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        'mean_clipped': (ratio > (1 + clip_ratio)).float().mean().item() + (ratio <
                                                                            (1 - clip_ratio)).float().mean().item(),
        'mean_advantage': advantages.mean().item(),
    }

    # Create return namedtuples
    loss_info = namedtuple('LossInfo', ['policy_loss'])(policy_loss=loss)
    metric_info = namedtuple('MetricInfo', list(metrics.keys()))(**metrics)

    return loss_info, metric_info
