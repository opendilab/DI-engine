from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .models import MLP, categorical_log_probs


@dataclass
class RSSMState:
    """确定性记忆 h、随机潜变量 z，以及构造 z 的分布参数。"""

    deter: torch.Tensor
    stoch: torch.Tensor
    stats: torch.Tensor


class RSSM(nn.Module):
    """递归状态空间模型。采集时用后验，想象时只用先验。"""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        *,
        stoch_classes: int = 0,
        unimix: float = 0.0
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.unimix = unimix
        self.stoch_feat_dim = stoch_dim * (stoch_classes or 1)
        v3 = bool(unimix)
        self.input_layer = MLP(self.stoch_feat_dim + action_dim, (), hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if v3 else nn.Identity()
        self.activation = nn.SiLU() if v3 else nn.ELU()
        self.rnn = nn.GRUCell(hidden_dim, deter_dim)
        out_dim = self.stoch_feat_dim if stoch_classes else 2 * stoch_dim
        self.prior_layer = MLP(deter_dim, (hidden_dim, ), out_dim, v3=v3)
        self.post_layer = MLP(deter_dim + embed_dim, (hidden_dim, ), out_dim, v3=v3)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_feat_dim, device=device)
        if self.stoch_classes:
            stats = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        else:
            stats = torch.cat([stoch, torch.ones_like(stoch)], dim=-1)
        return RSSMState(deter, stoch, stats)

    @staticmethod
    def get_feat(state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def recurrent(self, prev: RSSMState, action: torch.Tensor) -> torch.Tensor:
        """h_t = f(h_{t-1}, z_{t-1}, a_{t-1})。先验和后验共用 h_t。"""
        x = self.input_layer(torch.cat([prev.stoch, action], dim=-1))
        return self.rnn(self.activation(self.input_norm(x)), prev.deter)

    def _stats(self, raw: torch.Tensor) -> torch.Tensor:
        if self.stoch_classes:
            logits = raw.unflatten(-1, (self.stoch_dim, self.stoch_classes))
            return categorical_log_probs(logits, self.unimix)
        mean, raw_std = raw.chunk(2, dim=-1)
        return torch.cat([mean, F.softplus(raw_std) + 0.1], dim=-1)

    def _sample(self, stats: torch.Tensor) -> torch.Tensor:
        if self.stoch_classes:
            probs = stats.exp()
            # 指数竞赛等价于从每个类别分布独立采样，向量化处理各个随机因子。
            with torch.no_grad():
                index = (probs / torch.empty_like(probs).exponential_()).argmax(-1, keepdim=True)
                one_hot = torch.zeros_like(probs).scatter_(-1, index, 1)
            # 前向是硬 one-hot，反向通过 softmax 的概率。
            return (one_hot + (probs - probs.detach())).flatten(-2)
        mean, std = stats.chunk(2, dim=-1)
        return mean + std * torch.randn_like(mean)

    def img_step(self, prev: RSSMState, action: torch.Tensor) -> RSSMState:
        deter = self.recurrent(prev, action)
        stats = self._stats(self.prior_layer(deter))
        return RSSMState(deter, self._sample(stats), stats)

    def obs_step(self, deter: torch.Tensor, embed: torch.Tensor) -> RSSMState:
        stats = self._stats(self.post_layer(torch.cat([deter, embed], dim=-1)))
        return RSSMState(deter, self._sample(stats), stats)

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, state: RSSMState) -> Tuple[torch.Tensor, RSSMState]:
        """输入 [B,T,...]。actions[:, t] 是进入 obs[:, t] 的动作 a_{t-1}。"""
        posts = []
        for embed, action in zip(embeds.unbind(1), actions.unbind(1)):
            state = self.obs_step(self.recurrent(state, action), embed)
            posts.append(state)
        deter = torch.stack([s.deter for s in posts], dim=1)
        posterior = RSSMState(
            deter, torch.stack([s.stoch for s in posts], dim=1), torch.stack([s.stats for s in posts], dim=1)
        )
        # 观测序列只沿后验推进。先验这时只需分布参数，可以批量计算，省去无用的采样。
        return self._stats(self.prior_layer(deter)), posterior

    def kl_divergence(self, post: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """KL(q || p) 在潜变量维度求和，保留各样本和时间步。"""
        if self.stoch_classes:
            return (post.exp() * (post - prior)).sum((-1, -2))
        q_mean, q_std = post.chunk(2, dim=-1)
        p_mean, p_std = prior.chunk(2, dim=-1)
        return (
            (p_std.log() - q_std.log()) + 0.5 * ((q_std / p_std).square() + ((q_mean - p_mean) / p_std).square() - 1)
        ).sum(-1)
