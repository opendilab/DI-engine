from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DreamerConfig
from .models import Actor, MLP, SymexpTwoHot, symlog
from .rssm import RSSM, RSSMState


@contextmanager
def freeze_module_params(*modules: nn.Module):
    """冻结参数，保留对输入张量的梯度。用于 V1 的想象轨迹。"""
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    for param in params:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param in params:
            param.requires_grad_(True)


@dataclass
class TrainingMetrics:
    model_loss: float
    obs_loss: float
    reward_loss: float
    discount_loss: float
    kl_loss: float
    actor_loss: float
    critic_loss: float
    replay_value_loss: float


class WorldModel(nn.Module):
    """编码器、RSSM、观测重建、奖励和 continuation 预测。"""

    def __init__(self, config: DreamerConfig, version: str):
        super().__init__()
        self.config = config
        self.version = version
        v3 = version == "v3"
        self.free_nats = config.free_nats if config.free_nats is not None else {"v1": 3., "v2": 0., "v3": 1.}[version]
        self.rssm = RSSM(
            config.action_dim,
            config.embed_dim,
            config.deter_dim,
            config.stoch_dim,
            config.hidden_dim,
            stoch_classes=config.stoch_classes if version != "v1" else 0,
            unimix=config.unimix if v3 else 0.,
        )
        self.feat_dim = config.deter_dim + self.rssm.stoch_feat_dim
        width = config.hidden_dim
        self.encoder = MLP(config.obs_dim, (width, width), config.embed_dim, v3=v3)
        self.decoder = MLP(self.feat_dim, (width, width), config.obs_dim, v3=v3)
        self.reward_model = MLP(
            self.feat_dim, (width, width), config.bins if v3 else 1, v3=v3, output_scale=0. if v3 else 1.
        )
        self.discount_model = MLP(self.feat_dim, (width, width), 1, v3=v3)
        if v3:
            self.distribution = SymexpTwoHot(config.bins)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        # V3 的向量输入与重建目标使用相同的 symlog 坐标，不依赖运行中的均值/方差。
        return self.encoder(symlog(obs) if self.version == "v3" else obs)

    def kl_loss(self, post: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        kl = self.rssm.kl_divergence
        if self.version == "v1":
            return kl(post, prior).mean().clamp_min(self.free_nats)
        # 两项都是 KL(q || p)。动力学项拟合后验样本，表示项约束后验可预测。
        dynamics = kl(post.detach(), prior)
        representation = kl(post, prior.detach())
        if self.version == "v2":
            balance = self.config.kl_balance
            return (
                balance * dynamics.mean().clamp_min(self.free_nats) +
                (1 - balance) * representation.mean().clamp_min(self.free_nats)
            )
        # V3 对每个位置应用 free bits，再独立加权（Nature 版 β_dyn=1、β_rep=0.1）。
        return (
            self.config.dynamics_scale * dynamics.clamp_min(self.free_nats).mean() +
            self.config.representation_scale * representation.clamp_min(self.free_nats).mean()
        )

    def loss(
            self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor,
            init_state: RSSMState
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], RSSMState]:
        target_obs = symlog(obs) if self.version == "v3" else obs
        prior, posts = self.rssm.observe(self.encoder(target_obs), actions, init_state)
        feat = self.rssm.get_feat(posts)
        obs_loss = 0.5 * (self.decoder(feat) - target_obs).square().sum(-1).mean()
        pred_reward = self.reward_model(feat)
        if self.version == "v3":
            reward_loss = self.distribution.loss(pred_reward, rewards).mean()
        else:
            # 固定单位方差高斯的负对数似然，略去与参数无关的常数。
            reward_loss = 0.5 * F.mse_loss(pred_reward.squeeze(-1), rewards)
        discount_loss = F.binary_cross_entropy_with_logits(self.discount_model(feat).squeeze(-1), 1 - terminals)
        kl_loss = self.kl_loss(posts.stats, prior)
        loss = obs_loss + reward_loss + self.config.discount_scale * discount_loss + self.config.kl_scale * kl_loss
        return loss, (obs_loss, reward_loss, discount_loss, kl_loss), posts

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        raw = self.reward_model(feat)
        return self.distribution.mean(raw) if self.version == "v3" else raw.squeeze(-1)

    def predict_discount(self, feat: torch.Tensor) -> torch.Tensor:
        return self.config.discount * self.discount_model(feat).squeeze(-1).sigmoid()


def lambda_return(
        rewards: torch.Tensor, next_values: torch.Tensor, discounts: torch.Tensor, lambda_: float
) -> torch.Tensor:
    """沿时间轴逆推 G_t = r_{t+1} + d_{t+1}[(1-λ)V(s_{t+1}) + λG_{t+1}]。"""
    last = next_values[-1]
    returns = []
    for reward, value, discount in reversed(list(zip(rewards.unbind(), next_values.unbind(), discounts.unbind()))):
        last = reward + discount * torch.lerp(value, last, lambda_)
        returns.append(last)
    return torch.stack(returns[::-1])


def discount_cumprod(discounts: torch.Tensor) -> torch.Tensor:
    """动作 a_t 的训练权重是到达 s_t 的概率。当前动作的折扣从下一项开始乘。"""
    return torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]]).cumprod(0)


class DreamerV1Agent:
    """V1 使用高斯潜变量，并通过世界模型反传策略梯度。"""

    version = "v1"
    default_entropy = 0.0

    def __init__(self, config: DreamerConfig, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        self.entropy_scale = self.default_entropy if config.entropy_scale is None else config.entropy_scale
        self.world_model = WorldModel(config, self.version).to(device)
        feat_dim, width = self.world_model.feat_dim, config.hidden_dim
        v3 = self.version == "v3"
        self.actor = Actor(feat_dim, config.action_dim, width, unimix=config.unimix if v3 else 0.).to(device)
        self.critic = MLP(
            feat_dim, (width, width), config.bins if v3 else 1, v3=v3, output_scale=0. if v3 else 1.
        ).to(device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False) if self.version != "v1" else None
        self.model_opt = torch.optim.Adam(self.world_model.parameters(), lr=config.model_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self._updates = 0

    def init_state(self, batch_size: int) -> RSSMState:
        return self.world_model.rssm.init_state(batch_size, self.device)

    @torch.no_grad()
    def observe(self, obs: np.ndarray, state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).reshape(-1, self.config.obs_dim)
        embed = self.world_model.encode(obs_t)
        rssm = self.world_model.rssm
        return rssm.obs_step(rssm.recurrent(state, prev_action), embed)

    @torch.no_grad()
    def policy_actions(self, state: RSSMState, eval_mode: bool = False) -> torch.Tensor:
        dist = self.actor(self.world_model.rssm.get_feat(state))
        return dist.logits.argmax(-1) if eval_mode else dist.sample()

    def policy(self, state: RSSMState, eval_mode: bool = False) -> int:
        return int(self.policy_actions(state, eval_mode).item())

    def action_to_onehot(self, action: int) -> torch.Tensor:
        return F.one_hot(torch.tensor([action], device=self.device), self.config.action_dim).float()

    def train_step(self,
                   batch: Dict[str, np.ndarray],
                   *,
                   behavior: bool = True,
                   report: bool = True) -> Optional[TrainingMetrics]:
        obs = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        terminals = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)
        # reset 观测的动作编号是 -1，进入 RSSM 时转换为全零向量。
        actions_oh = (actions.unsqueeze(-1) == torch.arange(self.config.action_dim, device=self.device)).float()
        model_loss, model_metrics, posts = self.world_model.loss(
            obs, actions_oh, rewards, terminals, self.init_state(obs.shape[0])
        )
        modules_and_opts = [(self.world_model, self.model_opt)]
        actor_loss = critic_loss = replay_loss = model_loss.new_zeros(())
        if behavior and obs.shape[1] > 1:
            actor_loss, critic_loss, replay_loss = self._behavior_loss(posts, rewards, terminals)
            modules_and_opts.extend([(self.actor, self.actor_opt), (self.critic, self.critic_opt)])

        for _, opt in modules_and_opts:
            opt.zero_grad(set_to_none=True)
        (model_loss + actor_loss + critic_loss + self.config.replay_value_scale * replay_loss).backward()
        for module, opt in modules_and_opts:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_clip)
            opt.step()
        if len(modules_and_opts) > 1:
            self._updates += 1
            self._update_target()

        if report:
            # 一次传回所有标量，避免每项日志分别等待 GPU 同步。
            vals = torch.stack((model_loss, *model_metrics, actor_loss, critic_loss, replay_loss))
            return TrainingMetrics(*vals.detach().cpu().tolist())
        return None

    def _behavior_loss(self, posts: RSSMState, rewards: torch.Tensor,
                       terminals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 随机切片的首帧用来恢复时序上下文。其后的后验状态均可作为想象起点。
        start = RSSMState(*(x[:, 1:].detach().flatten(0, 1) for x in (posts.deter, posts.stoch, posts.stats)))
        alive = 1 - terminals[:, 1:].reshape(-1)
        return self._actor_critic_loss(start, alive, posts, rewards, terminals)

    def _imagine(self, state: RSSMState, *, pathwise: bool = False) -> Tuple[torch.Tensor, ...]:
        """返回 H+1 个状态特征和 H 个动作。从 s_t 采取 a_t 后预测 r_{t+1}、c_{t+1}。"""
        rssm = self.world_model.rssm
        features, actions, entropies = [rssm.get_feat(state)], [], []
        action_ids = torch.arange(self.config.action_dim, device=self.device)
        for _ in range(self.config.horizon):
            dist = self.actor(features[-1].detach())
            action = dist.sample()
            one_hot = (action.unsqueeze(-1) == action_ids).float()
            if pathwise:
                # 原 V1 针对连续控制。离散环境以直通 one-hot 保留同一条路径梯度。
                probs = dist.probs
                one_hot = one_hot + (probs - probs.detach())
                if self.entropy_scale:
                    entropies.append(dist.entropy())
            state = rssm.img_step(state, one_hot)
            features.append(rssm.get_feat(state))
            actions.append(action)
        return torch.stack(features), torch.stack(actions), entropies

    def _actor_critic_loss(
            self, start: RSSMState, alive: torch.Tensor, posts: RSSMState, rewards: torch.Tensor,
            terminals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 参数冻结期间，价值与奖励头仍能把梯度经由状态传回动作。
        with freeze_module_params(self.world_model, self.critic):
            feat, _, entropies = self._imagine(start, pathwise=True)
            reward = self.world_model.predict_reward(feat[1:])
            discount = self.world_model.predict_discount(feat[1:])
            values = self.critic(feat[1:]).squeeze(-1)
            returns = lambda_return(reward, values, discount, self.config.lambda_)
            weight = discount_cumprod(discount.detach()) * alive
            objective = returns
            if entropies:
                objective = objective + self.entropy_scale * torch.stack(entropies)
            actor_loss = -(weight * objective).mean()
        raw = self.critic(feat[:-1].detach()).squeeze(-1)
        critic_loss = 0.5 * (weight * (raw - returns.detach()).square()).mean()
        return actor_loss, critic_loss, critic_loss.new_zeros(())

    def _update_target(self) -> None:
        # V1 的 lambda-return 使用在线 critic。
        pass


class DreamerV2Agent(DreamerV1Agent):
    """V2 使用分类潜变量、KL balancing 和离散策略的 REINFORCE 目标。"""

    version = "v2"
    default_entropy = 2e-3

    def _critic_value(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.squeeze(-1)

    def _bootstrap_values(self, feat: torch.Tensor, online: torch.Tensor) -> torch.Tensor:
        return self._critic_value(self.critic_target(feat))

    def _advantage(self, returns: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        return returns - baseline

    def _value_loss(self, raw: torch.Tensor, target: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._critic_value(raw) - target).square()

    def _replay_loss(
            self, posts: RSSMState, rewards: torch.Tensor, terminals: torch.Tensor, starts_return: torch.Tensor
    ) -> torch.Tensor:
        return rewards.new_zeros(())

    def _actor_critic_loss(
            self, start: RSSMState, alive: torch.Tensor, posts: RSSMState, rewards: torch.Tensor,
            terminals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # V2（离散动作）和 V3 将想象轨迹当作数据，只对 log π(a_t|s_t) 和熵求策略梯度。
        with torch.no_grad():
            feat, action, _ = self._imagine(start)
            reward = self.world_model.predict_reward(feat[1:])
            discount = self.world_model.predict_discount(feat[1:])
            weight = discount_cumprod(discount) * alive
        online = self.critic(feat)
        with torch.no_grad():
            values = self._bootstrap_values(feat, online)
            returns = lambda_return(reward, values[1:], discount, self.config.lambda_)
            advantage = self._advantage(returns, values[:-1])
        dist = self.actor(feat[:-1])
        actor_loss = -(weight * (dist.log_prob(action) * advantage + self.entropy_scale * dist.entropy())).mean()
        critic_loss = (weight * self._value_loss(online[:-1], returns, feat[:-1])).mean()
        replay_loss = self._replay_loss(posts, rewards, terminals, returns[0])
        return actor_loss, critic_loss, replay_loss

    def _update_target(self) -> None:
        if self._updates % self.config.target_interval == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())


class DreamerV3Agent(DreamerV2Agent):
    """V3 使用 symlog、symexp two-hot、回报分位数及慢速 critic 正则。"""

    version = "v3"
    default_entropy = 3e-4

    def __init__(self, config: DreamerConfig, device: torch.device):
        super().__init__(config, device)
        self.return_bounds = torch.zeros(2, device=device)
        self.return_quantiles = torch.tensor([0.05, 0.95], device=device)

    def _critic_value(self, raw: torch.Tensor) -> torch.Tensor:
        return self.world_model.distribution.mean(raw)

    def _bootstrap_values(self, feat: torch.Tensor, online: torch.Tensor) -> torch.Tensor:
        # V3 用在线 critic 做 bootstrap。慢速 critic 只提供正则目标。
        return self._critic_value(online)

    def _advantage(self, returns: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        percentiles = torch.quantile(returns.reshape(-1), self.return_quantiles)
        self.return_bounds.lerp_(percentiles, self.config.return_ema)
        scale = (self.return_bounds[1] - self.return_bounds[0]).clamp_min(1.)
        return (returns - baseline) / scale

    def _value_loss(self, raw: torch.Tensor, target: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        dist = self.world_model.distribution
        with torch.no_grad():
            slow_value = self._critic_value(self.critic_target(feat))
        return dist.loss(raw, target) + self.config.slow_value_scale * dist.loss(raw, slow_value)

    def _replay_loss(
            self, posts: RSSMState, rewards: torch.Tensor, terminals: torch.Tensor, starts_return: torch.Tensor
    ) -> torch.Tensor:
        if rewards.shape[1] <= 2 or not self.config.replay_value_scale:
            return rewards.new_zeros(())
        # 真实后验序列的下一步奖励来自回放，bootstrap 来自该下一状态的想象回报。
        # 序列不会跨回合。真实终止置零折扣，时间上限和序列末端可 bootstrap。
        feat = self.world_model.rssm.get_feat(posts)[:, 1:-1]
        bootstrap = starts_return.reshape(rewards.shape[0], -1)[:, 1:].transpose(0, 1)
        target = lambda_return(
            rewards[:, 2:].transpose(0, 1), bootstrap, (1 - terminals[:, 2:].transpose(0, 1)) * self.config.discount,
            self.config.lambda_
        )
        # 该辅助损失也更新表示，使后验保留价值预测所需的信息。
        return self._value_loss(self.critic(feat), target.transpose(0, 1), feat).mean()

    @torch.no_grad()
    def _update_target(self) -> None:
        for param, slow_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            slow_param.lerp_(param, self.config.target_tau)


def make_agent(version: str, config: DreamerConfig, device: torch.device) -> DreamerV1Agent:
    version = version.lower()
    classes = {"v1": DreamerV1Agent, "v2": DreamerV2Agent, "v3": DreamerV3Agent}
    if version not in classes:
        raise ValueError(f"Unknown agent version: {version}")
    return classes[version](config, device)
