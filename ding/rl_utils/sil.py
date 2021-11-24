from collections import namedtuple
import torch
sil_data = namedtuple('sil_data', ['logit', 'action', 'value', 'total_reward'])


def sil_error(data: namedtuple):
    logit, action, value, total_reward = data
    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)
    adv = (total_reward - value).clamp(min=0.0)
    policy_loss = -logp * adv.detach()
    value_loss = 0.5 * (adv).pow(2)
    return policy_loss.mean(), value_loss.mean()
