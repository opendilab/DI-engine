from collections import namedtuple
from typing import Optional, Tuple
import torch

ppo_data = namedtuple('ppo_data', ['logp_new', 'logp_old', 'value_new', 'value_old', 'adv', 'return_'])
ppo_loss = namedtuple('ppo_loss', ['policy_loss', 'value_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])


def ppo_error(
        data: namedtuple,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lambda_: float = 0.97,
        use_value_clip: bool = True,
        dual_clip: Optional[float] = None
) -> Tuple[namedtuple, namedtuple]:
    """
        Shapes:
            - logp_new (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size
            - logp_old (:obj:`torch.FloatTensor`): :math:`(B, )`
            - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`
            - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
            - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
            - return_ (:obj:`torch.FloatTensor`): :math:`(B, )`
            - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
            - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        dual_clip: if used, default value is 5.0
    """
    logp_new, logp_old, value_new, value_old, adv, return_ = data
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        policy_loss = -torch.max(torch.min(surr1, surr2), dual_clip * adv).mean()
    else:
        policy_loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * torch.max(v1, v2).mean()
    else:
        value_loss = 0.5 * (return_ - value_new).pow(2).mean()

    return ppo_loss(policy_loss, value_loss), ppo_info(approx_kl, clipfrac)
