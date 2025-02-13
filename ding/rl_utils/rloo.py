from typing import Optional, Tuple
from collections import namedtuple
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

rloo_policy_data = namedtuple('rloo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight'])


def rloo_policy_error(
        data: namedtuple,
        clip_ratio: float = 0.2,
) -> Tuple[namedtuple, namedtuple]:

    """
    .. note::
        Each element in this input data is a group of response samples from the same prompt.
    """
    """计算RLOO (Rejection Learning with Optimistic Optimization) 的策略损失

       Args:
           data (rloo_policy_data): 包含以下字段的数据:
               - logit_new: 当前策略的logits [batch_size, seq_len, vocab_size]
               - logit_old: 旧策略的logits [batch_size, seq_len, vocab_size]
               - action: 实际采取的动作 [batch_size, seq_len]
               - adv: 优势值 [batch_size]
               - weight: 注意力掩码 [batch_size, seq_len]
           clip_ratio (float): PPO截断比率，默认0.2

       Returns:
           Tuple[namedtuple, namedtuple]:
               - 第一个namedtuple包含policy_loss
               - 第二个namedtuple包含额外的指标信息
       """
    # 计算每个token的log概率
    log_prob_new = torch.log_softmax(data.logit_new, dim=-1)
    log_prob_old = torch.log_softmax(data.logit_old, dim=-1)

    # 获取选定动作的log概率
    action = data.action.unsqueeze(-1)  # [batch_size, seq_len, 1]
    per_token_logps = torch.gather(log_prob_new, -1, action).squeeze(-1)  # [batch_size, seq_len]
    per_token_old_logps = torch.gather(log_prob_old, -1, action).squeeze(-1)

    # 计算策略比率
    ratio = torch.exp(per_token_logps - per_token_old_logps)
    ratio_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # 计算每个token的损失
    advantages = data.adv.unsqueeze(1)  # [batch_size, 1]
    per_token_loss_unclipped = ratio * advantages
    per_token_loss_clipped = ratio_clipped * advantages
    per_token_loss = -torch.min(per_token_loss_unclipped, per_token_loss_clipped)

    # 使用weight计算平均损失
    weight = data.weight if data.weight is not None else torch.ones_like(per_token_loss)
    loss = ((per_token_loss * weight).sum(dim=1) / weight.sum(dim=1)).mean()

    # 计算额外的指标
    metrics = {
        'mean_ratio': ((ratio * weight).sum(dim=1) / weight.sum(dim=1)).mean().item(),
        'mean_clipped': (ratio > (1 + clip_ratio)).float().mean().item() + \
                        (ratio < (1 - clip_ratio)).float().mean().item(),
        'mean_advantage': advantages.mean().item(),
    }

    # 创建返回的namedtuple对象
    loss_info = namedtuple('LossInfo', ['policy_loss'])(policy_loss=loss)
    metric_info = namedtuple('MetricInfo', list(metrics.keys()))(**metrics)

    return loss_info, metric_info


    raise NotImplementedError
