from typing import Tuple, List
from collections import namedtuple
import torch
import torch.nn.functional as F
EPS = 1e-8


def acer_policy_error(
        q_values: torch.Tensor,
        q_retraces: torch.Tensor,
        v_pred: torch.Tensor,
        target_logit: torch.Tensor,
        actions: torch.Tensor,
        ratio: torch.Tensor,
        c_clip_ratio: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overview:
        Get ACER policy loss
    Arguments:
        - q_values (:obj:`torch.Tensor`): Q values
        - q_retraces (:obj:`torch.Tensor`): Q values (be calculated by retrace method)
        - v_pred (:obj:`torch.Tensor`): V values
        - target_pi (:obj:`torch.Tensor`): The new policy's probability
        - actions (:obj:`torch.Tensor`): The actions in replay buffer
        - ratio (:obj:`torch.Tensor`): ratio of new polcy with behavior policy
        - c_clip_ratio (:obj:`float`): clip value for ratio
    Returns:
        - actor_loss (:obj:`torch.Tensor`): policy loss from q_retrace
        - bc_loss (:obj:`torch.Tensor`): correct policy loss
    Shapes:
        - q_values (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where B is batch size and N is action dim
        - q_retraces (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
        - v_pred (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
        - target_pi (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
        - ratio (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        - actor_loss (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
        - bc_loss (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
    """
    actions = actions.unsqueeze(-1)
    with torch.no_grad():
        advantage_retraces = q_retraces - v_pred  # shape T,B,1
        advantage_native = q_values - v_pred  # shape T,B,env_action_shape
    actor_loss = ratio.gather(-1, actions).clamp(max=c_clip_ratio) * advantage_retraces * target_logit.gather(
        -1, actions
    )  # shape T,B,1

    # bias correction term, the first target_pi will not calculate gradient flow
    bias_correction_loss = (1.0-c_clip_ratio/(ratio+EPS)).clamp(min=0.0)*torch.exp(target_logit).detach() * \
        advantage_native*target_logit  # shape T,B,env_action_shape
    bias_correction_loss = bias_correction_loss.sum(-1, keepdim=True)
    return actor_loss, bias_correction_loss


def acer_value_error(q_values, q_retraces, actions):
    """
    Overview:
        Get ACER critic loss
    Arguments:
        - q_values (:obj:`torch.Tensor`): Q values
        - q_retraces (:obj:`torch.Tensor`): Q values (be calculated by retrace method)
        - actions (:obj:`torch.Tensor`): The actions in replay buffer
        - ratio (:obj:`torch.Tensor`): ratio of new polcy with behavior policy
    Returns:
        - critic_loss (:obj:`torch.Tensor`): critic loss
    Shapes:
        - q_values (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where B is batch size and N is action dim
        - q_retraces (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
        - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
        - critic_loss (:obj:`torch.FloatTensor`): :math:`(T, B, 1)`
    """
    actions = actions.unsqueeze(-1)
    critic_loss = 0.5 * (q_retraces - q_values.gather(-1, actions)).pow(2)
    return critic_loss


def acer_trust_region_update(
        actor_gradients: List[torch.Tensor], target_logit: torch.Tensor, avg_logit: torch.Tensor,
        trust_region_value: float
) -> List[torch.Tensor]:
    """
    Overview:
        calcuate gradient with trust region constrain
    Arguments:
        - actor_gradients (:obj:`list(torch.Tensor)`): gradients value's for different part
        - target_pi (:obj:`torch.Tensor`): The new policy's probability
        - avg_pi (:obj:`torch.Tensor`): The average policy's probability
        - trust_region_value (:obj:`float`): the range of trust region
    Returns:
        - update_gradients (:obj:`list(torch.Tensor)`): gradients with trust region constraint
    Shapes:
        - target_pi (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
        - avg_pi (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
    """
    with torch.no_grad():
        KL_gradients = [torch.exp(avg_logit)]
    update_gradients = []
    # TODO: here is only one elements in this list.Maybe will use to more elements in the future
    actor_gradient = actor_gradients[0]
    KL_gradient = KL_gradients[0]
    scale = actor_gradient.mul(KL_gradient).sum(-1, keepdim=True) - trust_region_value
    scale = torch.div(scale, KL_gradient.mul(KL_gradient).sum(-1, keepdim=True)).clamp(min=0.0)
    update_gradients.append(actor_gradient - scale * KL_gradient)
    return update_gradients
