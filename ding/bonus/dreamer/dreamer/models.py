from typing import Iterable

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * x.abs().log1p()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * x.abs().expm1()


def categorical_log_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    if unimix:
        # V3 为每个类别保留 1% 均匀概率质量，避免 log-prob 和 KL 数值发散。
        return ((1 - unimix) * log_probs.exp() + unimix / logits.shape[-1]).log()
    return log_probs


class MLP(nn.Sequential):
    """向量观测版的编码器、解码器与标量预测头共用这个 MLP。"""

    def __init__(
        self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, *, v3: bool = False, output_scale: float = 1.0
    ):
        layers = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if v3:
                layers.extend([nn.LayerNorm(hidden), nn.SiLU()])
            else:
                layers.append(nn.ELU())
            in_dim = hidden
        output = nn.Linear(in_dim, out_dim)
        if output_scale != 1:
            with torch.no_grad():
                output.weight.mul_(output_scale)
                output.bias.mul_(output_scale)
        super().__init__(*layers, output)


class Actor(MLP):
    """离散策略 `π(a | h, z)`，计算策略梯度时保留采样所用的分布。"""

    def __init__(self, feat_dim: int, num_actions: int, hidden: int, *, unimix: float = 0.0):
        super().__init__(feat_dim, (hidden, hidden), num_actions, v3=bool(unimix), output_scale=0.01 if unimix else 1.0)
        self.unimix = unimix

    def forward(self, feat: torch.Tensor) -> Categorical:
        logits = super().forward(feat)
        if self.unimix:
            logits = categorical_log_probs(logits, self.unimix)
        return Categorical(logits=logits, validate_args=False)


class SymexpTwoHot(nn.Module):
    """在原始尺度上插值，在指数间隔的支撑点上拟合标量分布（V3）。"""

    def __init__(self, num_bins: int):
        super().__init__()
        if num_bins < 3 or num_bins % 2 == 0:
            raise ValueError("SymexpTwoHot requires an odd number of bins >= 3.")
        positive = symexp(torch.linspace(0, 20, num_bins // 2 + 1)[1:])
        self.register_buffer("bins", torch.cat([-positive.flip(0), torch.zeros(1), positive]))

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(-1)
        mid = self.bins.numel() // 2
        # 对称桶先配对。在均匀分布下，大幅值正负项能精确抵消，初始预测为零。
        return ((probs[..., mid + 1:] - probs[..., :mid].flip(-1)) * self.bins[mid + 1:]).sum(-1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().clamp(self.bins[0], self.bins[-1]).contiguous()
        upper = torch.searchsorted(self.bins, target).clamp(1, self.bins.numel() - 1)
        lower = upper - 1
        fraction = (target - self.bins[lower]) / (self.bins[upper] - self.bins[lower])
        log_probs = logits.log_softmax(-1)
        lo = log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        hi = log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        return -torch.lerp(lo, hi, fraction)
