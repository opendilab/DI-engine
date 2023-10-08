from typing import Tuple
from collections import namedtuple
import torch
import torch.nn.functional as F

ppg_data = namedtuple('ppg_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'])
ppg_joint_loss = namedtuple('ppg_joint_loss', ['auxiliary_loss', 'behavioral_cloning_loss'])


def ppg_joint_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
) -> Tuple[namedtuple, namedtuple]:
    '''
    Overview:
        Get PPG joint loss
    Arguments:
        - data (:obj:`namedtuple`): ppg input data with fieids shown in ``ppg_data``
        - clip_ratio (:obj:`float`): clip value for ratio
        - use_value_clip (:obj:`bool`): whether use value clip
    Returns:
        - ppg_joint_loss (:obj:`namedtuple`): the ppg loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B,)`
        - value_new (:obj:`torch.FloatTensor`): :math:`(B, 1)`
        - value_old (:obj:`torch.FloatTensor`): :math:`(B, 1)`
        - return_ (:obj:`torch.FloatTensor`): :math:`(B, 1)`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B,)`
        - auxiliary_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - behavioral_cloning_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> action_dim = 4
        >>> data = ppg_data(
        >>>     logit_new=torch.randn(3, action_dim),
        >>>     logit_old=torch.randn(3, action_dim),
        >>>     action=torch.randint(0, action_dim, (3,)),
        >>>     value_new=torch.randn(3, 1),
        >>>     value_old=torch.randn(3, 1),
        >>>     return_=torch.randn(3, 1),
        >>>     weight=torch.ones(3),
        >>> )
        >>> loss = ppg_joint_error(data, 0.99, 0.99)
    '''
    logit_new, logit_old, action, value_new, value_old, return_, weight = data

    if weight is None:
        weight = torch.ones_like(return_)

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
    behavioral_cloning_loss = F.kl_div(logp_new, logp_old, reduction='batchmean')

    return ppg_joint_loss(auxiliary_loss, behavioral_cloning_loss)
