import csv
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from tqdm import trange

try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym

from config import DreamerConfig, TrainConfig
from dreamer.agent import make_agent
from dreamer.replay import ReplayBuffer


def seed_everything(seed: int) -> None:
    """设置随机种子以提升可复现性。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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


class ObsNormalizer:
    """观测归一化器。"""

    def __init__(self, shape, clip: float = 5.0):
        """初始化观测归一化器。"""
        self.rms = RunningMeanStd(shape=shape)
        self.clip = clip

    def update(self, obs: np.ndarray) -> None:
        """更新观测统计量。"""
        self.rms.update(obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """对观测进行归一化并裁剪。"""
        obs = (obs - self.rms.mean) / (np.sqrt(self.rms.var) + 1e-8)
        return np.clip(obs, -self.clip, self.clip)


def reset_env(env, seed=None):
    """重置环境并兼容 gym/gymnasium 返回格式。"""
    out = env.reset(seed=seed)
    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out
    return obs


def step_env(env, action):
    """执行一步交互并统一返回格式。"""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        terminal = terminated
    else:
        obs, reward, done, info = out
        terminal = done
    return obs, reward, done, terminal, info


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    """按训练步数计算 epsilon 值。"""
    if cfg.exploration_decay_steps <= 0:
        return cfg.exploration_epsilon_end
    progress = min(1.0, step / float(cfg.exploration_decay_steps))
    return cfg.exploration_epsilon_start + progress * (cfg.exploration_epsilon_end - cfg.exploration_epsilon_start)


def evaluate(agent, env_id: str, episodes: int, seed: int, normalizer: Optional[ObsNormalizer], env_kwargs=None) -> float:
    """评估当前策略的平均回报。"""
    env = gym.make(env_id, **(env_kwargs or {}))
    scores = []
    for ep in range(episodes):
        obs = reset_env(env, seed=seed + 1000 + ep)
        state = agent.init_state(1)
        prev_action = torch.zeros(1, agent.config.action_dim, device=agent.device)
        done = False
        total = 0.0
        while not done:
            obs_in = normalizer(obs) if normalizer is not None else obs
            state = agent.observe(obs_in, state, prev_action)
            action = agent.policy(state, eval_mode=True)
            prev_action = agent.action_to_onehot(action)
            obs, reward, done, _, _ = step_env(env, action)
            total += reward
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def init_metrics_csv(path: str, overwrite: bool = False) -> None:
    """初始化评估指标 CSV 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_steps", "avg_return", "seed"])


def append_metrics_csv(path: str, env_steps: int, avg_return: float, seed: int) -> None:
    """追加一条评估记录。"""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([env_steps, f"{avg_return:.4f}", seed])


def reset_episode(env, agent, action_dim: int, seed: Optional[int] = None):
    """重置环境并初始化状态。"""
    obs = reset_env(env, seed=seed)
    state = agent.init_state(1)
    prev_action = torch.zeros(1, action_dim, device=agent.device)
    return obs, state, prev_action


def normalize_obs(obs: np.ndarray, normalizer: Optional[ObsNormalizer]) -> np.ndarray:
    """按需更新并归一化观测。"""
    if normalizer is None:
        return obs
    normalizer.update(np.asarray([obs]))
    return normalizer(obs)


def resolve_run_workdir(base_workdir: str, time_format: str, timestamp_if_exists: bool) -> str:
    """若基础目录已存在，则追加时间戳形成新的实验目录。"""
    if (not timestamp_if_exists) or (not os.path.exists(base_workdir)):
        return base_workdir
    timestamp = time.strftime(time_format, time.localtime())
    candidate = f"{base_workdir}_{timestamp}"
    if not os.path.exists(candidate):
        return candidate
    suffix = 1
    while True:
        candidate_with_suffix = f"{candidate}_{suffix}"
        if not os.path.exists(candidate_with_suffix):
            return candidate_with_suffix
        suffix += 1


def train(cfg: TrainConfig, env_id: str, version: str, seed: int) -> None:
    """训练单个环境/版本/种子组合。"""
    seed_everything(seed)

    # 1. 初始化实验目录
    seed_dir = os.path.join(cfg.workdir, env_id, version, f"seed_{seed}")
    metrics_csv = os.path.join(seed_dir, "metrics", "eval_returns.csv")
    os.makedirs(seed_dir, exist_ok=True)
    init_metrics_csv(metrics_csv, overwrite=cfg.overwrite_metrics)

    # 2. 初始化环境与基础信息
    env = gym.make(env_id, **(cfg.env_kwargs or {}))
    obs = reset_env(env, seed=seed)
    if not hasattr(env.action_space, "n"):
        raise ValueError(f"{env_id} action space is not discrete; this implementation only supports Discrete.")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # 3. 初始化 Agent / Replay / Normalizer
    config = DreamerConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        embed_dim=cfg.embed_dim,
        deter_dim=cfg.deter_dim,
        stoch_dim=cfg.stoch_dim,
        stoch_classes=cfg.stoch_classes,
        hidden_dim=cfg.hidden_dim,
        horizon=cfg.horizon,
        model_lr=cfg.model_lr,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        entropy_scale=cfg.entropy_scale,
        use_obs_norm=cfg.use_obs_norm,
        normalize_advantage=cfg.normalize_advantage,
        discount=cfg.discount,
        lambda_=cfg.lambda_,
        free_nats=cfg.free_nats,
        kl_scale=cfg.kl_scale,
        discount_scale=cfg.discount_scale,
        kl_balance=cfg.kl_balance,
        target_tau=cfg.target_tau,
    )
    device = torch.device(cfg.device)
    agent = make_agent(version, config, device)

    replay = ReplayBuffer(cfg.replay_size)
    normalizer = ObsNormalizer(shape=obs.shape) if config.use_obs_norm else None

    episode_obs, episode_actions, episode_rewards, episode_dones = [], [], [], []
    episode_return = 0.0
    recent_returns = deque(maxlen=10)

    obs, state, prev_action = reset_episode(env, agent, action_dim, seed=seed)

    pbar = trange(cfg.total_steps, desc=f"Training {version} {env_id} seed {seed}")
    for step in pbar:
        # 4. 观测预处理
        obs_raw = obs
        obs_norm = normalize_obs(obs_raw, normalizer)

        # 5. 更新后验状态
        state = agent.observe(obs_norm, state, prev_action)

        # 6. 选择动作（探索 or 策略）
        if step < cfg.seed_steps:
            action = env.action_space.sample()
        else:
            eps = epsilon_by_step(step - cfg.seed_steps, cfg)
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = agent.policy(state, eval_mode=False)

        # 7. 与环境交互
        prev_action = agent.action_to_onehot(action)
        next_obs, reward, done, _, _ = step_env(env, action)

        # 8. 记录 episode 数据
        # 这里保存动作后的观测，使 obs / action / reward / done 在时间上严格对齐。
        episode_obs.append(next_obs)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_dones.append(done)
        episode_return += reward

        obs = next_obs

        # 9. Episode 结束处理
        if done:
            replay.add_episode(episode_obs, episode_actions, episode_rewards, episode_dones)
            recent_returns.append(episode_return)
            obs, state, prev_action = reset_episode(env, agent, action_dim, seed=seed + step + 1)
            episode_obs, episode_actions, episode_rewards, episode_dones = [], [], [], []
            episode_return = 0.0

        # 10. 训练更新
        if len(replay) >= cfg.batch_size * cfg.seq_len and step >= cfg.seed_steps:
            if step % cfg.train_every == 0:
                metrics = None
                for _ in range(cfg.train_steps):
                    batch = replay.sample(cfg.batch_size, cfg.seq_len)
                    if normalizer is not None:
                        batch["obs"] = normalizer(batch["obs"])
                    metrics = agent.train_step(batch)
                if metrics is not None:
                    pbar.set_postfix(
                        {
                            "model": f"{metrics.model_loss:.3f}",
                            "actor": f"{metrics.actor_loss:.3f}",
                            "critic": f"{metrics.critic_loss:.3f}",
                            "return": f"{np.mean(recent_returns) if recent_returns else 0.0:.1f}",
                        }
                    )

        # 11. 日志打印
        if (step + 1) % cfg.log_every == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            print(f"Step {step + 1} | avg_return={avg_return:.1f}")

        # 12. 定期评估 + 记录 CSV
        if (step + 1) % cfg.eval_every == 0:
            eval_return = evaluate(agent, env_id, cfg.eval_episodes, seed, normalizer, env_kwargs=cfg.env_kwargs)
            print(f"Eval at step {step + 1}: avg_return={eval_return:.1f}")
            append_metrics_csv(metrics_csv, step + 1, eval_return, seed)

    env.close()


def main() -> None:
    """入口函数，依次训练所有组合并绘图。"""
    cfg = TrainConfig()
    cfg.workdir = resolve_run_workdir(
        cfg.workdir,
        time_format=cfg.workdir_time_format,
        timestamp_if_exists=cfg.timestamp_workdir_if_exists,
    )
    cfg.plot_path = os.path.join(cfg.workdir, "return_curve.png")
    os.makedirs(cfg.workdir, exist_ok=True)
    print(f"Run workdir: {cfg.workdir}")

    for env_id in cfg.env_ids:
        for version in cfg.agent_versions:
            for seed in cfg.seeds:
                print(f"\n=== Training {version} on {env_id} with seed {seed} ===")
                train(cfg, env_id, version, seed)

    try:
        from plot_returns import plot_returns

        plot_returns(workdir=cfg.workdir, plot_path=cfg.plot_path, agent_versions=cfg.agent_versions)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
