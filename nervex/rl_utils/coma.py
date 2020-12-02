from collections import namedtuple
import torch
import torch.nn.functional as F
from nervex.torch_utils import one_hot

coma_data = namedtuple('coma_data', ['logit', 'action', 'q_val', 'adv', 'return_', 'weight', 'mask'])
coma_loss = namedtuple('coma_loss', ['policy_loss', 'q_val_loss', 'entropy_loss'])


def coma_error(data: namedtuple) -> namedtuple:
    """
    Overview:
        Implementation of COMA
    Arguments:
        - data (:obj:`namedtuple`): coma input data with fieids shown in ``coma_data``
    Returns:
        - coma_loss (:obj:`namedtuple`): the coma loss item, all of them are the differentiable 0-dim tensor
    Shapes:
        - logit (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`, where B is batch size A is the agent num, and N is \
            action dim
        - action (:obj:`torch.LongTensor`): :math:`(T, B, A)`
        - q_val (:obj:`torch.FloatTensor`): :math:`(T, B, A, N)`
        - adv (:obj:`torch.FloatTensor`): :math:`(T, B, A)`
        - return (:obj:`torch.FloatTensor`): :math:`(T-1, B, A)`
        - mask (:obj:`torch.LongTensor`): :math:`(T, B, A)`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(T ,B, A)`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    """
    logit, action, q_val, adv, return_, weight, mask = data
    if weight is None:
        weight = torch.ones_like(action)
    action_dim = logit.shape[3]
    action_onehot = one_hot(action, action_dim)
    dist = torch.distributions.categorical.Categorical(logits=logit)
    logp = dist.log_prob(action)
    q_taken = torch.sum(q_val * action_onehot, -1)
    entropy_loss = (dist.entropy() * weight).sum() / mask.sum()
    policy_loss = -(logp * adv * weight).sum() / mask.sum()
    q_val_loss = (F.mse_loss(return_, q_taken[:-1], reduction='none') * weight[:-1]).sum() / mask[:-1].sum()
    return coma_loss(policy_loss, q_val_loss, entropy_loss)
