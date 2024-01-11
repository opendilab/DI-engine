from collections import namedtuple
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from ding.hpc_rl import hpc_wrapper

happo_value_data = namedtuple('happo_value_data', ['value_new', 'value_old', 'return_', 'weight'])
happo_loss = namedtuple('happo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
happo_policy_loss = namedtuple('happo_policy_loss', ['policy_loss', 'entropy_loss'])
happo_info = namedtuple('happo_info', ['approx_kl', 'clipfrac'])
happo_data = namedtuple(
    'happo_data', ['logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'adv', 'return_', 'weight', 'factor']
)
happo_policy_data = namedtuple('happo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight', 'factor'])


def happo_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
        dual_clip: Optional[float] = None,
) -> Tuple[namedtuple, namedtuple]:
    """
    Overview:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip
    Arguments:
        - data (:obj:`namedtuple`): the ppo input data with fieids shown in ``ppo_data``
        - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2
        - use_value_clip (:obj:`bool`): whether to use clip in value loss with the same ratio as policy
        - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
        defaults to 5.0, if you don't want to use it, set this parameter to None
    Returns:
        - happo_loss (:obj:`namedtuple`): the ppo loss item, all of them are the differentiable 0-dim tensor
        - happo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
    Shapes:
        - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`
        - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> action_dim = 4
        >>> data = happo_data(
        >>>     logit_new=torch.randn(3, action_dim),
        >>>     logit_old=torch.randn(3, action_dim),
        >>>     action=torch.randint(0, action_dim, (3,)),
        >>>     value_new=torch.randn(3),
        >>>     value_old=torch.randn(3),
        >>>     adv=torch.randn(3),
        >>>     return_=torch.randn(3),
        >>>     weight=torch.ones(3),
        >>>     factor=torch.ones(3, 1),
        >>> )
        >>> loss, info = happo_error(data)

    .. note::

        adv is already normalized value (adv - adv.mean()) / (adv.std() + 1e-8), and there are many
        ways to calculate this mean and std, like among data buffer or train batch, so we don't couple
        this part into happo_error, you can refer to our examples for different ways.
    """
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    logit_new, logit_old, action, value_new, value_old, adv, return_, weight, factor = data
    policy_data = happo_policy_data(logit_new, logit_old, action, adv, weight, factor)
    policy_output, policy_info = happo_policy_error(policy_data, clip_ratio, dual_clip)
    value_data = happo_value_data(value_new, value_old, return_, weight)
    value_loss = happo_value_error(value_data, clip_ratio, use_value_clip)

    return happo_loss(policy_output.policy_loss, value_loss, policy_output.entropy_loss), policy_info


def happo_policy_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        dual_clip: Optional[float] = None,
) -> Tuple[namedtuple, namedtuple]:
    '''
    Overview:
        Get PPO policy loss
    Arguments:
        - data (:obj:`namedtuple`): ppo input data with fieids shown in ``ppo_policy_data``
        - clip_ratio (:obj:`float`): clip value for ratio
        - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
        defaults to 5.0, if you don't want to use it, set this parameter to None
    Returns:
        - happo_policy_loss (:obj:`namedtuple`): the ppo policy loss item, all of them are the differentiable \
            0-dim tensor.
        - happo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
    Shapes:
        - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> action_dim = 4
        >>> data = ppo_policy_data(
        >>>     logit_new=torch.randn(3, action_dim),
        >>>     logit_old=torch.randn(3, action_dim),
        >>>     action=torch.randint(0, action_dim, (3,)),
        >>>     adv=torch.randn(3),
        >>>     weight=torch.ones(3),
        >>>     factor=torch.ones(3, 1),
        >>> )
        >>> loss, info = happo_policy_error(data)
    '''
    logit_new, logit_old, action, adv, weight, factor = data
    if weight is None:
        weight = torch.ones_like(adv)
    dist_new = torch.distributions.categorical.Categorical(logits=logit_new)
    dist_old = torch.distributions.categorical.Categorical(logits=logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    dist_new_entropy = dist_new.entropy()
    if dist_new_entropy.shape != weight.shape:
        dist_new_entropy = dist_new.entropy().mean(dim=1)
    entropy_loss = (dist_new_entropy * weight).mean()
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    if ratio.shape != adv.shape:
        ratio = ratio.mean(dim=1)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    # shape factor: (B,1)  surr1: (B,)
    clip1 = torch.min(surr1, surr2) * factor.squeeze(1)
    if dual_clip is not None:
        clip2 = torch.max(clip1, dual_clip * adv)
        # only use dual_clip when adv < 0
        policy_loss = -(torch.where(adv < 0, clip2, clip1) * weight).mean()
    else:
        policy_loss = (-clip1 * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    return happo_policy_loss(policy_loss, entropy_loss), happo_info(approx_kl, clipfrac)


def happo_value_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
) -> torch.Tensor:
    '''
    Overview:
        Get PPO value loss
    Arguments:
        - data (:obj:`namedtuple`): ppo input data with fieids shown in ``happo_value_data``
        - clip_ratio (:obj:`float`): clip value for ratio
        - use_value_clip (:obj:`bool`): whether use value clip
    Returns:
        - value_loss (:obj:`torch.FloatTensor`): the ppo value loss item, \
            all of them are the differentiable 0-dim tensor
    Shapes:
        - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size
        - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
    Examples:
        >>> action_dim = 4
        >>> data = happo_value_data(
        >>>     value_new=torch.randn(3),
        >>>     value_old=torch.randn(3),
        >>>     return_=torch.randn(3),
        >>>     weight=torch.ones(3),
        >>> )
        >>> loss, info = happo_value_error(data)
    '''
    value_new, value_old, return_, weight = data
    if weight is None:
        weight = torch.ones_like(value_old)
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        value_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()
    return value_loss


def happo_error_continuous(
        data: namedtuple,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
        dual_clip: Optional[float] = None,
) -> Tuple[namedtuple, namedtuple]:
    """
    Overview:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with value_clip and dual_clip
    Arguments:
        - data (:obj:`namedtuple`): the ppo input data with fieids shown in ``ppo_data``
        - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2
        - use_value_clip (:obj:`bool`): whether to use clip in value loss with the same ratio as policy
        - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
        defaults to 5.0, if you don't want to use it, set this parameter to None
    Returns:
        - happo_loss (:obj:`namedtuple`): the ppo loss item, all of them are the differentiable 0-dim tensor
        - happo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
    Shapes:
        - mu_sigma_new (:obj:`tuple`): :math:`((B, N), (B, N))`, where B is batch size and N is action dim
        - mu_sigma_old (:obj:`tuple`): :math:`((B, N), (B, N))`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - value_new (:obj:`torch.FloatTensor`): :math:`(B, )`
        - value_old (:obj:`torch.FloatTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - return (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - value_loss (:obj:`torch.FloatTensor`): :math:`()`
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> action_dim = 4
        >>> data = ppo_data_continuous(
        >>>     mu_sigma_new= dict(mu=torch.randn(3, action_dim), sigma=torch.randn(3, action_dim)**2),
        >>>     mu_sigma_old= dict(mu=torch.randn(3, action_dim), sigma=torch.randn(3, action_dim)**2),
        >>>     action=torch.randn(3, action_dim),
        >>>     value_new=torch.randn(3),
        >>>     value_old=torch.randn(3),
        >>>     adv=torch.randn(3),
        >>>     return_=torch.randn(3),
        >>>     weight=torch.ones(3),
        >>> )
        >>> loss, info = happo_error(data)

    .. note::

        adv is already normalized value (adv - adv.mean()) / (adv.std() + 1e-8), and there are many
        ways to calculate this mean and std, like among data buffer or train batch, so we don't couple
        this part into happo_error, you can refer to our examples for different ways.
    """
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    mu_sigma_new, mu_sigma_old, action, value_new, value_old, adv, return_, weight, factor_batch = data
    if weight is None:
        weight = torch.ones_like(adv)

    dist_new = Normal(mu_sigma_new['mu'], mu_sigma_new['sigma'])
    if len(mu_sigma_old['mu'].shape) == 1:
        dist_old = Normal(mu_sigma_old['mu'].unsqueeze(-1), mu_sigma_old['sigma'].unsqueeze(-1))
    else:
        dist_old = Normal(mu_sigma_old['mu'], mu_sigma_old['sigma'])
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    entropy_loss = (dist_new.entropy() * weight.unsqueeze(1)).mean()

    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    ratio = torch.prod(ratio, dim=-1)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        # shape factor: (B,1)  surr1: (B,)
        policy_loss = (-torch.max(factor_batch.squeeze(1) * torch.min(surr1, surr2), dual_clip * adv) * weight).mean()
    else:
        policy_loss = (-factor_batch.squeeze(1) * torch.min(surr1, surr2) * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # value_loss
    if use_value_clip:
        value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
        v1 = (return_ - value_new).pow(2)
        v2 = (return_ - value_clip).pow(2)
        value_loss = 0.5 * (torch.max(v1, v2) * weight).mean()
    else:
        value_loss = 0.5 * ((return_ - value_new).pow(2) * weight).mean()

    return happo_loss(policy_loss, value_loss, entropy_loss), happo_info(approx_kl, clipfrac)


def happo_policy_error_continuous(data: namedtuple,
                                  clip_ratio: float = 0.2,
                                  dual_clip: Optional[float] = None) -> Tuple[namedtuple, namedtuple]:
    """
    Overview:
        Implementation of Proximal Policy Optimization (arXiv:1707.06347) with dual_clip
    Arguments:
        - data (:obj:`namedtuple`): the ppo input data with fieids shown in ``ppo_data``
        - clip_ratio (:obj:`float`): the ppo clip ratio for the constraint of policy update, defaults to 0.2
        - dual_clip (:obj:`float`): a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
        defaults to 5.0, if you don't want to use it, set this parameter to None
    Returns:
        - happo_loss (:obj:`namedtuple`): the ppo loss item, all of them are the differentiable 0-dim tensor
        - happo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
    Shapes:
        - mu_sigma_new (:obj:`tuple`): :math:`((B, N), (B, N))`, where B is batch size and N is action dim
        - mu_sigma_old (:obj:`tuple`): :math:`((B, N), (B, N))`, where B is batch size and N is action dim
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
        >>> action_dim = 4
        >>> data = ppo_policy_data_continuous(
        >>>     mu_sigma_new=dict(mu=torch.randn(3, action_dim), sigma=torch.randn(3, action_dim)**2),
        >>>     mu_sigma_old=dict(mu=torch.randn(3, action_dim), sigma=torch.randn(3, action_dim)**2),
        >>>     action=torch.randn(3, action_dim),
        >>>     adv=torch.randn(3),
        >>>     weight=torch.ones(3),
        >>> )
        >>> loss, info = happo_policy_error_continuous(data)
    """
    assert dual_clip is None or dual_clip > 1.0, "dual_clip value must be greater than 1.0, but get value: {}".format(
        dual_clip
    )
    mu_sigma_new, mu_sigma_old, action, adv, weight = data
    if weight is None:
        weight = torch.ones_like(adv)

    dist_new = Independent(Normal(mu_sigma_new['mu'], mu_sigma_new['sigma']), 1)
    if len(mu_sigma_old['mu'].shape) == 1:
        dist_old = Independent(Normal(mu_sigma_old['mu'].unsqueeze(-1), mu_sigma_old['sigma'].unsqueeze(-1)), 1)
    else:
        dist_old = Independent(Normal(mu_sigma_old['mu'], mu_sigma_old['sigma']), 1)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    entropy_loss = (dist_new.entropy() * weight).mean()
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        policy_loss = (-torch.max(torch.min(surr1, surr2), dual_clip * adv) * weight).mean()
    else:
        policy_loss = (-torch.min(surr1, surr2) * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    return happo_policy_loss(policy_loss, entropy_loss), happo_info(approx_kl, clipfrac)
