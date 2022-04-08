import torch
import torch.nn.functional as F
from collections import namedtuple
from .isw import compute_importance_weights


def compute_q_retraces(
        q_values: torch.Tensor,
        v_pred: torch.Tensor,
        rewards: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        ratio: torch.Tensor,
        gamma: float = 0.9
) -> torch.Tensor:
    """
    Shapes:
        - q_values (:obj:`torch.Tensor`): :math:`(T + 1, B, N)`, where T is unroll_len, B is batch size, N is discrete \
            action dim.
        - v_pred (:obj:`torch.Tensor`): :math:`(T + 1, B, 1)`
        - rewards (:obj:`torch.Tensor`): :math:`(T, B)`
        - actions (:obj:`torch.Tensor`): :math:`(T, B)`
        - weights (:obj:`torch.Tensor`): :math:`(T, B)`
        - ratio (:obj:`torch.Tensor`): :math:`(T, B, N)`
        - q_retraces (:obj:`torch.Tensor`): :math:`(T + 1, B, 1)`

    .. note::
        q_retrace operation doesn't need to compute gradient, just executes forward computation.
    """
    T = q_values.size()[0] - 1
    rewards = rewards.unsqueeze(-1)
    actions = actions.unsqueeze(-1)
    weights = weights.unsqueeze(-1)
    q_retraces = torch.zeros_like(v_pred)  # shape (T+1),B,1
    tmp_retraces = v_pred[-1]  # shape B,1
    q_retraces[-1] = v_pred[-1]

    q_gather = torch.zeros_like(v_pred)
    q_gather[0:-1] = q_values[0:-1].gather(-1, actions)  # shape (T+1),B,1
    ratio_gather = ratio.gather(-1, actions)  # shape T,B,1

    for idx in reversed(range(T)):
        q_retraces[idx] = rewards[idx] + gamma * weights[idx] * tmp_retraces
        tmp_retraces = ratio_gather[idx].clamp(max=1.0) * (q_retraces[idx] - q_gather[idx]) + v_pred[idx]
    return q_retraces  # shape (T+1),B,1
