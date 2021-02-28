from collections import namedtuple
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal

ppg_data = namedtuple('ppg_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'])
ppg_aux_data = namedtuple('ppg_aux_data', ['value_new', 'value_old', 'return_', 'weight'])
ppg_joint_loss = namedtuple('ppg_joint_loss', ['auxiliary_loss', 'behavioral_cloning_loss'])
ppg_aux_loss = namedtuple('ppg_aux_loss', ['auxiliary_loss', 'behavioral_cloning_loss', 'value_loss'])


def ppg_joint_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
) -> Tuple[namedtuple, namedtuple]:
    logit_new, logit_old, action, value_new, value_old, return_, weight = data

    # auxiliary_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        auxiliary_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        auxiliary_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()

    dist_new = torch.distributions.categorical.Categorical(logits=logit_new)
    dist_old = torch.distributions.categorical.Categorical(logits=logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)

    # behavioral cloning loss
    approx_kl = (logp_old - logp_new).mean()
    behavioral_cloning_loss = F.kl_div(logp_new, logp_old, reduction='batchmean')

    return ppg_joint_loss(auxiliary_loss, behavioral_cloning_loss)


def value_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
) -> Tuple[namedtuple, namedtuple]:
    value_new, value_old, return_, weight = data
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        value_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()
    return {'value_loss': value_loss}
