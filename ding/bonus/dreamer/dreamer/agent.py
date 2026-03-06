from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import DreamerConfig
from .models import Actor, Critic, Decoder, DiscountModel, Encoder, RewardModel
from .rssm import RSSM, RSSMState


def symlog(x: torch.Tensor) -> torch.Tensor:
    """对称对数变换，压缩数值范围。"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symlog 的逆变换。"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def two_hot(x: torch.Tensor, bins: int, min_value: float, max_value: float) -> torch.Tensor:
    """将标量转成 two-hot 分布表示。"""
    x = torch.clamp(x, min=min_value, max=max_value)
    scale = (bins - 1) / (max_value - min_value)
    pos = (x - min_value) * scale
    idx_low = torch.floor(pos).long()
    idx_high = torch.clamp(idx_low + 1, max=bins - 1)
    weight_high = (pos - idx_low.float()).clamp(0.0, 1.0)
    weight_low = 1.0 - weight_high
    shape = x.shape + (bins,)
    out = torch.zeros(shape, device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx_low.unsqueeze(-1), weight_low.unsqueeze(-1))
    out.scatter_add_(-1, idx_high.unsqueeze(-1), weight_high.unsqueeze(-1))
    return out


def logits_to_mean(logits: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """将分布 logits 转成对应的期望值。"""
    probs = torch.softmax(logits, dim=-1)
    bins = torch.linspace(min_value, max_value, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return torch.sum(probs * bins, dim=-1)


@contextmanager
def freeze_module_params(*modules: Optional[nn.Module]):
    """临时冻结模块参数，但保留对输入的梯度传播。"""
    params = []
    requires_grad = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            params.append(param)
            requires_grad.append(param.requires_grad)
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, flag in zip(params, requires_grad):
            param.requires_grad_(flag)


class RunningMeanStd:
    """在线估计均值与方差，用于归一化。"""

    def __init__(self, epsilon: float = 1e-4, shape=()):
        """初始化统计量。"""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """用新样本更新均值与方差。"""
        x = np.asarray(x, dtype="float64")
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        """从批次统计量更新。"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


@dataclass
class TrainingMetrics:
    """记录一次训练更新中的关键指标。"""

    model_loss: float
    obs_loss: float
    reward_loss: float
    discount_loss: float
    kl_loss: float
    actor_loss: float
    critic_loss: float


class WorldModel(nn.Module):
    """世界模型：编码器 + RSSM + 重建/奖励/折扣预测。"""
    def __init__(self, config: DreamerConfig,
        kl_balance: Optional[float] = None, *,
        discrete_latent: bool = False,
    ):
        """初始化世界模型的子模块与配置。"""
        super().__init__()
        self.encoder = Encoder(config.obs_dim, config.embed_dim, config.hidden_dim)
        self.rssm = RSSM(
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            hidden_dim=config.hidden_dim,
            discrete=discrete_latent,
            stoch_classes=config.stoch_classes,
        )
        stoch_feat_dim = config.stoch_dim * config.stoch_classes if discrete_latent else config.stoch_dim
        feat_dim = config.deter_dim + stoch_feat_dim
        self.decoder = Decoder(feat_dim, config.obs_dim, config.hidden_dim)
        reward_out_dim = config.reward_bins if config.reward_bins and config.reward_bins > 1 else 1
        self.reward_model = RewardModel(feat_dim, config.hidden_dim, out_dim=reward_out_dim)
        self.discount_model = DiscountModel(feat_dim, config.hidden_dim)
        self.config = config
        self.kl_balance = kl_balance
        self.reward_bins = reward_out_dim
        self.reward_min = config.reward_min
        self.reward_max = config.reward_max
        self.use_symlog = config.use_symlog

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, init_state: RSSMState) -> Tuple[torch.Tensor, Dict[str, float], RSSMState]:
        """计算世界模型损失，并返回后验状态序列。"""
        embeds = self.encoder(obs)
        priors, posts = self.rssm.observe(embeds, actions, init_state)
        feat = self.rssm.get_feat(posts)
        obs_pred = self.decoder(feat)
        reward_pred = self.reward_model(feat)
        discount_logits = self.discount_model(feat).squeeze(-1)

        obs_loss = F.mse_loss(obs_pred, obs)
        reward_loss = self.reward_loss(reward_pred, rewards)
        discount_target = 1.0 - dones
        discount_loss = F.binary_cross_entropy_with_logits(discount_logits, discount_target)

        kl_post = self.rssm.kl_divergence(posts, priors)
        if self.kl_balance is None:
            kl = kl_post
        else:
            # DreamerV2/V3 的 balanced KL 使用同方向 KL，并分别停止 posterior / prior 的梯度。
            post_sg = self.rssm.detach_state(posts)
            prior_sg = self.rssm.detach_state(priors)
            kl_prior = self.rssm.kl_divergence(post_sg, priors)
            kl_post = self.rssm.kl_divergence(posts, prior_sg)
            kl = self.kl_balance * kl_prior + (1.0 - self.kl_balance) * kl_post
        kl_loss = torch.mean(torch.clamp(kl, min=self.config.free_nats))

        model_loss = obs_loss + reward_loss + self.config.discount_scale * discount_loss + self.config.kl_scale * kl_loss
        metrics = {
            "model_loss": model_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "discount_loss": discount_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return model_loss, metrics, posts

    def reward_loss(self, reward_pred: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """奖励预测损失（可选 two-hot + symlog）。"""
        if self.reward_bins == 1:
            return F.mse_loss(reward_pred.squeeze(-1), rewards)
        target = symlog(rewards) if self.use_symlog else rewards
        target = two_hot(target, self.reward_bins, self.reward_min, self.reward_max)
        log_probs = F.log_softmax(reward_pred, dim=-1)
        return -(target * log_probs).sum(-1).mean()

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        """将奖励头输出转为标量奖励。"""
        pred = self.reward_model(feat)
        if self.reward_bins == 1:
            return pred.squeeze(-1)
        mean = logits_to_mean(pred, self.reward_min, self.reward_max)
        return symexp(mean) if self.use_symlog else mean


def lambda_return(rewards: torch.Tensor, values: torch.Tensor, discounts: torch.Tensor, bootstrap: torch.Tensor, lambda_: float) -> torch.Tensor:
    """计算 lambda-return。"""
    horizon = rewards.shape[0]
    returns = []
    last = bootstrap
    for t in reversed(range(horizon)):
        next_value = values[t + 1] if t + 1 < horizon else bootstrap
        last = rewards[t] + discounts[t] * ((1 - lambda_) * next_value + lambda_ * last)
        returns.append(last)
    returns.reverse()
    return torch.stack(returns, dim=0)


def discount_cumprod(discounts: torch.Tensor) -> torch.Tensor:
    """计算折扣累计乘积，用于重要性权重。"""
    horizon = discounts.shape[0]
    weights = []
    acc = torch.ones_like(discounts[0])
    for t in range(horizon):
        weights.append(acc)
        acc = acc * discounts[t]
    return torch.stack(weights, dim=0)


class DreamerV1Agent:
    """Simplified DreamerV1 baseline."""

    def __init__(self, config: DreamerConfig, device: torch.device):
        """初始化 DreamerV1 的模型与优化器。"""
        self.config = config
        self.device = device
        self.world_model = self._build_world_model().to(device)
        stoch_feat_dim = config.stoch_dim * config.stoch_classes if self._discrete_latent() else config.stoch_dim
        feat_dim = config.deter_dim + stoch_feat_dim
        self.actor = Actor(feat_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic = Critic(feat_dim, config.hidden_dim, out_dim=self._critic_out_dim()).to(device)
        self.critic_target = self._build_target_critic(feat_dim, self._critic_out_dim())

        self.model_opt = torch.optim.Adam(self.world_model.parameters(), lr=config.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def init_state(self, batch_size: int) -> RSSMState:
        """初始化 RSSM 隐状态。"""
        return self.world_model.rssm.init_state(batch_size, self.device)

    def observe(self, obs: np.ndarray, state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        """使用观测更新后验状态。"""
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            prior = self.world_model.rssm.img_step(state, prev_action)
            post = self.world_model.rssm.obs_step(prior.deter, embed)
        return post

    def policy(self, state: RSSMState, eval_mode: bool = False) -> int:
        """根据当前状态采样/选择动作。"""
        feat = self.world_model.rssm.get_feat(state)
        dist = self.actor(feat)
        if eval_mode:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    def action_to_onehot(self, action: int) -> torch.Tensor:
        """将离散动作转换为 one-hot。"""
        return F.one_hot(torch.tensor([action], device=self.device), num_classes=self.config.action_dim).float()

    def train_step(self, batch: Dict[str, np.ndarray]) -> TrainingMetrics:
        """执行一次训练更新（世界模型 + actor/critic）。"""
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)

        actions_oh = F.one_hot(actions, num_classes=self.config.action_dim).float()

        init_state = self.init_state(obs.shape[0])
        model_loss, model_metrics, posts = self.world_model.loss(obs, actions_oh, rewards, dones, init_state)

        self.model_opt.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.model_opt.step()

        actor_loss, critic_loss = self._actor_critic_update(posts, dones)

        return TrainingMetrics(
            model_loss=model_metrics["model_loss"],
            obs_loss=model_metrics["obs_loss"],
            reward_loss=model_metrics["reward_loss"],
            discount_loss=model_metrics["discount_loss"],
            kl_loss=model_metrics["kl_loss"],
            actor_loss=actor_loss,
            critic_loss=critic_loss,
        )

    def _actor_critic_update(self, posts: RSSMState, dones: torch.Tensor) -> Tuple[float, float]:
        """在想象轨迹上更新策略与价值网络。"""
        start = self.world_model.rssm.detach_state(posts)
        if start.deter.shape[1] <= 1:
            return 0.0, 0.0
        # chunk 的第一个 posterior 缺少前文上下文，跳过它能减轻 RSSM 冷启动偏差。
        start = RSSMState(
            deter=start.deter[:, 1:],
            stoch=start.stoch[:, 1:],
            stats=start.stats[:, 1:],
        )
        dones = dones[:, 1:]
        # flatten batch and time
        b, t, _ = start.deter.shape
        nonterminal = (dones.reshape(b * t) < 0.5)
        if not torch.any(nonterminal):
            return 0.0, 0.0
        start = RSSMState(
            deter=start.deter.reshape(b * t, -1)[nonterminal],
            stoch=start.stoch.reshape(b * t, -1)[nonterminal],
            stats=start.stats.reshape(b * t, *start.stats.shape[2:])[nonterminal],
        )

        with freeze_module_params(self.world_model, self.critic, self.critic_target):
            feats, entropies, rewards, discounts, last_state = self._imagine(start)
            values_raw = self._critic_raw(feats)
            values = self._critic_value(values_raw)
            bootstrap_raw = self._critic_raw(self.world_model.rssm.get_feat(last_state))
            bootstrap = self._critic_value(bootstrap_raw)
            returns = lambda_return(rewards, values, discounts, bootstrap, self.config.lambda_)
            weights = discount_cumprod(discounts).detach()
            actor_loss = -(weights * returns).mean() - self.config.entropy_scale * (weights * entropies).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_opt.step()

        feats_detached = feats.detach()
        with torch.no_grad():
            target_raw = self._target_critic_raw(feats_detached)
            target_values = self._critic_value(target_raw)
            last_feat = self.world_model.rssm.get_feat(last_state).detach()
            bootstrap_raw = self._target_critic_raw(last_feat)
            bootstrap = self._critic_value(bootstrap_raw)
            returns = lambda_return(rewards.detach(), target_values, discounts.detach(), bootstrap, self.config.lambda_)
            weights = discount_cumprod(discounts.detach())

        values_raw = self._critic_raw(feats_detached)
        critic_loss = self._critic_loss(values_raw, returns, weights)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_opt.step()
        self._update_target()

        return actor_loss.item(), critic_loss.item()

    def _imagine(self, start: RSSMState):
        """在世界模型中进行想象 rollout。"""
        feats = []
        entropies = []
        rewards = []
        discounts = []
        state = start
        for _ in range(self.config.horizon):
            feat = self.world_model.rssm.get_feat(state)
            dist = self.actor(feat)
            action = dist.sample()
            entropy = dist.entropy()
            action_oh = F.one_hot(action, num_classes=self.config.action_dim).float()
            action_oh = action_oh + dist.probs - dist.probs.detach()

            state = self.world_model.rssm.img_step(state, action_oh)
            next_feat = self.world_model.rssm.get_feat(state)
            reward = self.world_model.predict_reward(next_feat)
            discount = torch.sigmoid(self.world_model.discount_model(next_feat).squeeze(-1)) * self.config.discount

            feats.append(feat)
            entropies.append(entropy)
            rewards.append(reward)
            discounts.append(discount)

        feats = torch.stack(feats, dim=0)
        entropies = torch.stack(entropies, dim=0)
        rewards = torch.stack(rewards, dim=0)
        discounts = torch.stack(discounts, dim=0)

        return feats, entropies, rewards, discounts, state

    def _update_target(self) -> None:
        """Polyak 平滑更新目标 critic。"""
        if self.critic_target is None:
            return
        tau = self.config.target_tau
        if tau <= 0.0:
            return
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * p.data)

    def _build_world_model(self) -> WorldModel:
        """构建世界模型（V1 为连续潜在）。"""
        return WorldModel(self.config, kl_balance=None, discrete_latent=self._discrete_latent())

    def _build_target_critic(self, feat_dim: int, out_dim: int):
        """构建目标 critic，用于稳定 bootstrap。"""
        critic_target = Critic(feat_dim, self.config.hidden_dim, out_dim=out_dim).to(self.device)
        critic_target.load_state_dict(self.critic.state_dict())
        return critic_target

    def _critic_out_dim(self) -> int:
        """critic 输出维度（V1 为标量）。"""
        return 1

    def _critic_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """critic 原始输出。"""
        return self.critic(feats)

    def _critic_value(self, raw: torch.Tensor) -> torch.Tensor:
        """将 critic 输出转换为标量价值。"""
        return raw.squeeze(-1)

    def _target_critic_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """目标 critic 的原始输出。"""
        if self.critic_target is None:
            return self._critic_raw(feats)
        return self.critic_target(feats)

    def _compute_advantages(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """计算优势函数。"""
        return returns - values

    def _critic_loss(self, raw_values: torch.Tensor, returns: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """critic 的回归损失。"""
        values = self._critic_value(raw_values)
        return ((values - returns.detach()) ** 2 * weights).mean()

    def _discrete_latent(self) -> bool:
        """是否使用离散潜在（V1 为 False）。"""
        return False
    
    
class DreamerV2Agent(DreamerV1Agent):
    """Simplified DreamerV2: KL balancing."""

    def _build_world_model(self) -> WorldModel:
        """V2 使用 KL 平衡的世界模型。"""
        return WorldModel(self.config, kl_balance=self.config.kl_balance, discrete_latent=self._discrete_latent())

    def _discrete_latent(self) -> bool:
        """V2 使用离散潜在表示。"""
        return True


class DreamerV3Agent(DreamerV2Agent):
    """Simplified DreamerV3: symlog + two-hot value/reward + target critic."""

    def __init__(self, config: DreamerConfig, device: torch.device):
        """启用 symlog、two-hot 回归与目标 critic。"""
        if config.reward_bins <= 1:
            config.reward_bins = 255
        if config.value_bins <= 1:
            config.value_bins = 255
        config.use_symlog = True
        super().__init__(config, device)
        self.return_rms = RunningMeanStd(shape=())

    def _critic_out_dim(self) -> int:
        """V3 critic 输出为分布 logits。"""
        return self.config.value_bins

    def _build_world_model(self) -> WorldModel:
        """V3 仍使用离散潜在 + KL 平衡。"""
        return WorldModel(self.config, kl_balance=self.config.kl_balance, discrete_latent=self._discrete_latent())

    def _build_target_critic(self, feat_dim: int, out_dim: int):
        """构建目标 critic，用于稳定训练。"""
        critic_target = Critic(feat_dim, self.config.hidden_dim, out_dim=out_dim).to(self.device)
        critic_target.load_state_dict(self.critic.state_dict())
        return critic_target

    def _critic_value(self, raw: torch.Tensor) -> torch.Tensor:
        """将 two-hot logits 转为标量价值。"""
        symlog_value = logits_to_mean(raw, self.config.value_min, self.config.value_max)
        return symexp(symlog_value)

    def _target_critic_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """目标 critic 原始输出。"""
        return self.critic_target(feats)

    def _compute_advantages(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """对回报做归一化后计算优势。"""
        if not self.config.normalize_advantage:
            return returns - values
        returns_np = returns.detach().cpu().numpy().reshape(-1)
        self.return_rms.update(returns_np)
        mean = torch.tensor(self.return_rms.mean, device=returns.device, dtype=returns.dtype)
        std = torch.tensor(np.sqrt(self.return_rms.var) + 1e-8, device=returns.device, dtype=returns.dtype)
        returns_norm = (returns - mean) / std
        values_norm = (values - mean) / std
        return returns_norm - values_norm

    def _critic_loss(self, raw_values: torch.Tensor, returns: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """V3 的 two-hot 价值回归损失。"""
        target = symlog(returns)
        target = two_hot(target, self.config.value_bins, self.config.value_min, self.config.value_max)
        log_probs = F.log_softmax(raw_values, dim=-1)
        loss = -(target * log_probs).sum(-1)
        return (loss * weights).mean()

    def _update_target(self) -> None:
        """Polyak 平滑更新目标 critic。"""
        super()._update_target()


def make_agent(version: str, config: DreamerConfig, device: torch.device):
    """根据版本字符串构造对应的 Agent。"""
    version = version.lower()
    if version == "v1":
        return DreamerV1Agent(config, device)
    if version == "v2":
        return DreamerV2Agent(config, device)
    if version == "v3":
        return DreamerV3Agent(config, device)
    raise ValueError(f"Unknown agent version: {version}")
