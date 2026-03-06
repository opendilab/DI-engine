from typing import Iterable, Optional

import torch
from torch import nn
from torch.distributions import Categorical


class MLP(nn.Module):
    """多层感知机模块。"""
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, act=nn.ELU, out_act: Optional[nn.Module] = None):
        """构建 MLP。"""
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-1], out_dim))
        if out_act is not None:
            layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.net(x)


class Encoder(nn.Module):
    """观测编码器。"""
    def __init__(self, obs_dim: int, embed_dim: int, hidden: int):
        """构建编码器。"""
        super().__init__()
        self.net = MLP(obs_dim, [hidden, hidden], embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """将观测编码为嵌入向量。"""
        return self.net(obs)


class Decoder(nn.Module):
    """观测解码器。"""
    def __init__(self, feat_dim: int, obs_dim: int, hidden: int):
        """构建解码器。"""
        super().__init__()
        self.net = MLP(feat_dim, [hidden, hidden], obs_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """从特征重建观测。"""
        return self.net(feat)


class RewardModel(nn.Module):
    """奖励预测模型。"""
    def __init__(self, feat_dim: int, hidden: int, out_dim: int = 1):
        """构建奖励模型。"""
        super().__init__()
        self.net = MLP(feat_dim, [hidden, hidden], out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """预测奖励（或分布 logits）。"""
        return self.net(feat)


class DiscountModel(nn.Module):
    """折扣（终止）预测模型。"""
    def __init__(self, feat_dim: int, hidden: int):
        """构建折扣模型。"""
        super().__init__()
        self.net = MLP(feat_dim, [hidden, hidden], 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """预测折扣 logits。"""
        return self.net(feat)


class Actor(nn.Module):
    """策略网络，输出动作分布。"""
    def __init__(self, feat_dim: int, num_actions: int, hidden: int):
        """构建策略网络。"""
        super().__init__()
        self.net = MLP(feat_dim, [hidden, hidden], num_actions)
        self.num_actions = num_actions

    def forward(self, feat: torch.Tensor) -> Categorical:
        """输出动作分布。"""
        logits = self.net(feat)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """价值网络。"""
    def __init__(self, feat_dim: int, hidden: int, out_dim: int = 1):
        """构建价值网络。"""
        super().__init__()
        self.net = MLP(feat_dim, [hidden, hidden], out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """输出价值（或分布 logits）。"""
        return self.net(feat)
