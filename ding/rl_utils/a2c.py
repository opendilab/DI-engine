from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal

a2c_data = namedtuple('a2c_data', ['logit', 'action', 'value', 'adv', 'return_', 'weight'])
a2c_loss = namedtuple('a2c_loss', ['policy_loss', 'value_loss', 'entropy_loss'])


def a2c_error(data: namedtuple) -> namedtuple:
    """
    Overview:
        Implementation of A2C(Advantage Actor-Critic) (arXiv:1602.01783) for discrete action space
    Arguments:
        - data (:obj:`namedtuple`): a2c input data with fieids shown in ``a2c_data``
    Returns:
        - a2c_loss (:obj:`namedtuple`): the a2c loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - value (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> data = a2c_data(
        >>>     logit=torch.randn(2, 3),
        >>>     action=torch.randint(0, 3, (2, )),
        >>>     value=torch.randn(2, ),
        >>>     adv=torch.randn(2, ),
        >>>     return_=torch.randn(2, ),
        >>>     weight=torch.ones(2, ),
        >>> )
        >>> loss = a2c_error(data)
    """
    logit, action, value, adv, return_, weight = data
    if weight is None:
        weight = torch.ones_like(value)
    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)
    entropy_loss = (dist.entropy() * weight).mean()
    policy_loss = -(logp * adv * weight).mean()
    value_loss = (F.mse_loss(return_, value, reduction='none') * weight).mean()
    return a2c_loss(policy_loss, value_loss, entropy_loss)


def a2c_error_continuous(data: namedtuple) -> namedtuple:
    """
    Overview:
        Implementation of A2C(Advantage Actor-Critic) (arXiv:1602.01783) for continuous action space
    Arguments:
        - data (:obj:`namedtuple`): a2c input data with fieids shown in ``a2c_data``
    Returns:
        - a2c_loss (:obj:`namedtuple`): the a2c loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, N)`
        - value (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> data = a2c_data(
        >>>     logit={'mu': torch.randn(2, 3), 'sigma': torch.sqrt(torch.randn(2, 3)**2)},
        >>>     action=torch.randn(2, 3),
        >>>     value=torch.randn(2, ),
        >>>     adv=torch.randn(2, ),
        >>>     return_=torch.randn(2, ),
        >>>     weight=torch.ones(2, ),
        >>> )
        >>> loss = a2c_error_continuous(data)
    """
    logit, action, value, adv, return_, weight = data
    if weight is None:
        weight = torch.ones_like(value)

    dist = Independent(Normal(logit['mu'], logit['sigma']), 1)
    logp = dist.log_prob(action)
    entropy_loss = (dist.entropy() * weight).mean()
    policy_loss = -(logp * adv * weight).mean()
    value_loss = (F.mse_loss(return_, value, reduction='none') * weight).mean()
    return a2c_loss(policy_loss, value_loss, entropy_loss)
