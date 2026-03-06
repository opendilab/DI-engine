from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F


@dataclass
class RSSMState:
    """RSSM 隐状态容器：同时包含确定性部分(h)与随机部分(z)。"""
    deter: torch.Tensor       # h_t (GRU state)
    stoch: torch.Tensor       # z_t (Sampled latent)
    stats: torch.Tensor       # Mean/Std (V1) 或 Logits (V2/V3)

class RSSM(nn.Module):
    """Recurrent State Space Model，支持连续/离散潜在。"""
    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int, 
    stoch_dim: int, hidden_dim: int, min_std: float = 0.1,
    discrete: bool = False, stoch_classes: int = 0,
    ):
        """初始化 RSSM 结构与参数。"""
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        self.discrete = discrete
        self.stoch_classes = stoch_classes
        if self.discrete and self.stoch_classes <= 1:
            raise ValueError("Discrete RSSM requires stoch_classes > 1.")

        stoch_input = stoch_dim * stoch_classes if self.discrete else stoch_dim
        self.input_layer = nn.Linear(stoch_input + action_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, deter_dim)
        if self.discrete:
            self.prior_layer = nn.Linear(deter_dim, stoch_dim * stoch_classes)
            self.post_layer = nn.Linear(deter_dim + embed_dim, stoch_dim * stoch_classes)
        else:
            self.prior_layer = nn.Linear(deter_dim, 2 * stoch_dim)
            self.post_layer = nn.Linear(deter_dim + embed_dim, 2 * stoch_dim)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """创建初始隐状态。"""
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        if self.discrete:
            stoch = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=device)
            stats = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        else:
            stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
            mean = torch.zeros(batch_size, self.stoch_dim, device=device)
            std = torch.ones(batch_size, self.stoch_dim, device=device)
            stats = torch.cat([mean, std], dim=-1)
        return RSSMState(deter=deter, stoch=stoch, stats=stats)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        """拼接确定性与随机性状态作为特征。"""
        return torch.cat([state.deter, state.stoch], dim=-1)

    def _stats(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """将线性输出拆分为 mean/std。"""
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """连续潜在采样。"""
        return mean + std * torch.randn_like(mean)

    def _sample_discrete(self, logits: torch.Tensor) -> torch.Tensor:
        """离散潜在采样（straight-through）。"""
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample()
        probs = dist.probs
        return sample + probs - probs.detach()

    def img_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        """先验步：仅依赖前一状态与动作。"""
        x = torch.cat([prev_state.stoch, action], dim=-1)
        x = F.elu(self.input_layer(x))
        deter = self.rnn(x, prev_state.deter)
        if self.discrete:
            logits = self.prior_layer(deter).view(-1, self.stoch_dim, self.stoch_classes)
            stoch = self._sample_discrete(logits).reshape(-1, self.stoch_dim * self.stoch_classes)
            return RSSMState(deter=deter, stoch=stoch, stats=logits)
        mean, std = self._stats(self.prior_layer(deter))
        stoch = self._sample(mean, std)
        stats = torch.cat([mean, std], dim=-1)
        return RSSMState(deter=deter, stoch=stoch, stats=stats)

    def obs_step(self, deter: torch.Tensor, embed: torch.Tensor) -> RSSMState:
        """后验步：融合观测嵌入。"""
        x = torch.cat([deter, embed], dim=-1)
        if self.discrete:
            logits = self.post_layer(x).view(-1, self.stoch_dim, self.stoch_classes)
            stoch = self._sample_discrete(logits).reshape(-1, self.stoch_dim * self.stoch_classes)
            return RSSMState(deter=deter, stoch=stoch, stats=logits)
        mean, std = self._stats(self.post_layer(x))
        stoch = self._sample(mean, std)
        stats = torch.cat([mean, std], dim=-1)
        return RSSMState(deter=deter, stoch=stoch, stats=stats)

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, init_state: RSSMState) -> (RSSMState, RSSMState):
        """按时间序列滚动，返回 prior 与 posterior 序列。"""
        priors: List[RSSMState] = []
        posts: List[RSSMState] = []
        state = init_state
        time_steps = embeds.shape[1]
        for t in range(time_steps):
            prior = self.img_step(state, actions[:, t])
            post = self.obs_step(prior.deter, embeds[:, t])
            state = post
            priors.append(prior)
            posts.append(post)
        return stack_states(priors), stack_states(posts)

    def detach_state(self, state: RSSMState) -> RSSMState:
        """从计算图中分离隐状态。"""
        return RSSMState(
            deter=state.deter.detach(),
            stoch=state.stoch.detach(),
            stats=state.stats.detach(),
        )

    def kl_divergence(self, post: RSSMState, prior: RSSMState) -> torch.Tensor:
        """计算 posterior 与 prior 的 KL。"""
        if self.discrete:
            post_logits = post.stats
            prior_logits = prior.stats
            post_dist = OneHotCategorical(logits=post_logits)
            prior_dist = OneHotCategorical(logits=prior_logits)
            kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
            return torch.sum(kl, dim=-1)
        post_mean, post_std = torch.chunk(post.stats, 2, dim=-1)
        prior_mean, prior_std = torch.chunk(prior.stats, 2, dim=-1)
        post_dist = Normal(post_mean, post_std)
        prior_dist = Normal(prior_mean, prior_std)
        kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        return torch.sum(kl, dim=-1)


def stack_states(states: List[RSSMState], dim: int = 1) -> RSSMState:
    """按时间维堆叠状态列表。"""
    return RSSMState(
        deter=torch.stack([s.deter for s in states], dim=dim),
        stoch=torch.stack([s.stoch for s in states], dim=dim),
        stats=torch.stack([s.stats for s in states], dim=dim),
    )
