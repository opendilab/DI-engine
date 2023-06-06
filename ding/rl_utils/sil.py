from collections import namedtuple
import torch
import torch.nn.functional as F

sil_data = namedtuple('sil_data', ['logit', 'action', 'value', 'adv', 'return_', 'weight'])
sil_loss = namedtuple('sil_loss', ['policy_loss', 'value_loss'])
sil_info = namedtuple('sil_info', ['policy_clipfrac', 'value_clipfrac'])


def sil_error(data: namedtuple) -> namedtuple:
    """
    Overview:
        Implementation of SIL(Self-Imitation Learning) (arXiv:1806.05635)
    Arguments:
        - data (:obj:`namedtuple`): SIL input data with fields shown in ``sil_data``
    Returns:
        - sil_loss (:obj:`namedtuple`): the SIL loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - value (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
    """
    logit, action, value, adv, return_, weight = data
    if weight is None:
        weight = torch.ones_like(value)
    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)

    # Clip the negative part of adv.
    policy_clipfrac = adv.lt(0).float().mean().item()
    adv = adv.clamp_min(0)
    policy_loss = -(logp * adv * weight).mean()

    # Clip the negative part of the distance between value and return.
    rv_dist = return_ - value
    value_clipfrac = rv_dist.lt(0).float().mean().item()
    rv_dist = rv_dist.clamp_min(0)
    value_loss = (F.mse_loss(rv_dist, torch.zeros_like(rv_dist), reduction='none') * weight).mean()
    return sil_loss(policy_loss, value_loss), sil_info(policy_clipfrac, value_clipfrac)
