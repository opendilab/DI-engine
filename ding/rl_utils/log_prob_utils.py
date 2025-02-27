from typing import List, Callable, Optional, Any
import torch
from torch import Tensor

LogitsProcessor = Callable[[Tensor, Tensor], Tensor]


def naive_method(logits: Tensor, index: Tensor) -> Tensor:
    """Calculate per-token log probabilities using naive method.

    Args:
        logits: Token logits of shape [B, S, V] or [S, V] where:
               B = batch size
               S = sequence length
               V = vocabulary size
        index: Selected token indices of shape [B, S] or [S]

    Returns:
        Tensor: Log probabilities for selected tokens of shape [B, S] or [S]
    """
    # Calculate log probabilities for each token
    log_prob_new: Tensor = torch.log_softmax(logits, dim=-1)
    # Get log probabilities for selected actions
    index = index.unsqueeze(-1)  # [B, S, 1] or [S, 1]
    per_token_logps: Tensor = torch.gather(log_prob_new, -1, index).squeeze(-1)
    return per_token_logps


def efficient_method(logits: Tensor, index: Tensor) -> Tensor:
    """Calculate per-token log probabilities efficiently.

    Args:
        logits: Token logits of shape [B, S, V] or [S, V] where:
               B = batch size
               S = sequence length
               V = vocabulary size
        index: Selected token indices of shape [B, S] or [S]

    Returns:
        Tensor: Log probabilities for selected tokens of shape [B, S] or [S]
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits: Tensor = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

        # Loop to reduce peak mem consumption
        logsumexp_values: Tensor = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])

        # log_softmax(x_i) = x_i - logsumexp(x)
        per_token_logps: Tensor = selected_logits - logsumexp_values
    else:
        # logsumexp approach is unstable with bfloat16
        per_token_logps: List[Tensor] = []

        # Loop to reduce peak mem consumption
        for row_logits, row_labels in zip(logits, index):  # Iterate over sequence length
            row_logps: Tensor = torch.log_softmax(row_logits, dim=-1)
            row_per_token_logps: Tensor = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)

        per_token_logps = torch.stack(per_token_logps)

    return per_token_logps


def less_efficient_method(logits: Tensor, index: Tensor) -> Tensor:
    """Calculate per-token log probabilities using categorical distribution.

    Args:
        logits: Token logits of shape [B, S, V] or [S, V] where:
               B = batch size
               S = sequence length
               V = vocabulary size
        index: Selected token indices of shape [B, S] or [S]

    Returns:
        Tensor: Log probabilities for selected tokens of shape [B, S] or [S]
    """
    dist = torch.distributions.categorical.Categorical(logits=logits)
    logp: Tensor = dist.log_prob(index)
    return logp


# 定义一个统一的类型
LogProbFunction = Callable[[Tensor, Tensor], Tensor]

# 导出所有方法
__all__ = ['naive_method', 'efficient_method', 'less_efficient_method', 'LogProbFunction']
