from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch


@dataclass
class DreamerConfig:
    """Dreamer 全局配置：涵盖模型结构、优化器参数与算法开关。"""
    # --- 基础环境参数 ---
    obs_dim: int
    action_dim: int
    
    # --- RSSM 核心维度 ---
    embed_dim: int = 64        # 观测编码后的维度
    deter_dim: int = 128       # 确定性状态 h_t 的维度 (GRU)
    stoch_dim: int = 32        # 随机状态 z_t 的维度
    stoch_classes: int = 32    # (V2/V3) 离散潜在变量的类别数
    hidden_dim: int = 128      # MLP 隐藏层维度
    
    # --- 训练超参数 ---
    model_lr: float = 3e-4     # 世界模型学习率
    actor_lr: float = 8e-5     # 策略网络学习率
    critic_lr: float = 8e-5    # 价值网络学习率
    entropy_scale: float = 1e-3
    grad_clip: float = 100.0   # 梯度裁剪阈值
    use_obs_norm: bool = True
    normalize_advantage: bool = True
    
    # --- RL 算法参数 ---
    discount: float = 0.99     # 折扣因子 gamma
    lambda_: float = 0.95      # Lambda-return 平滑系数
    horizon: int = 15          # 想象视界长度 H
    
    # --- Loss 权重 ---
    free_nats: float = 1.0     # KL 散度的 Free bits 阈值
    kl_scale: float = 1.0      # KL Loss 权重
    discount_scale: float = 10.0
    
    # --- V2 特性开关 ---
    kl_balance: float = 0.8    # KL Balancing 权重 (0.8 给先验)
    
    # --- V3 特性开关 ---
    use_symlog: bool = False   # 是否启用 Symlog 数值压缩
    target_tau: float = 0.01   # Critic Target 软更新系数
    
    # --- V3 离散回归配置 ---
    reward_bins: int = 0       # 奖励离散化的桶数量 (0表示使用标量回归)
    reward_min: float = -10.0
    reward_max: float = 10.0
    value_bins: int = 0        # 价值离散化的桶数量
    value_min: float = -20.0
    value_max: float = 20.0


@dataclass
class TrainConfig:
    """训练流程相关配置。"""
    env_ids: Sequence[str] = ("CartPole-v1",)
    agent_versions: Sequence[str] = ("v1",)
    seeds: Sequence[int] = (42, 2024)
    total_steps: int = 30_000
    seed_steps: int = 2_000
    train_every: int = 1
    train_steps: int = 1
    batch_size: int = 32
    seq_len: int = 8
    horizon: int = 15
    replay_size: int = 100_000
    log_every: int = 1000
    eval_every: int = 2_000
    eval_episodes: int = 5
    workdir: str = "runs/dreamer"
    workdir_time_format: str = "%Y%m%d_%H%M%S"
    timestamp_workdir_if_exists: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env_kwargs: Optional[Dict[str, Any]] = None

    # Exploration schedule (epsilon-greedy after seed steps)
    exploration_epsilon_start: float = 0.10
    exploration_epsilon_end: float = 0.00
    exploration_decay_steps: int = 20_000

    # Actor regularization / reward scaling
    entropy_scale: float = 1e-3
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Model configuration
    embed_dim: int = 64
    deter_dim: int = 128
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 128
    use_obs_norm: bool = True
    normalize_advantage: bool = True
    discount: float = 0.99
    lambda_: float = 0.95
    free_nats: float = 1.0
    kl_scale: float = 1.0
    discount_scale: float = 10.0
    kl_balance: float = 0.8
    target_tau: float = 0.01

    # 是否每次训练前清空 metrics 文件，避免重复运行时混入历史点
    overwrite_metrics: bool = True

    # Plotting
    plot_path: str = "runs/dreamer/return_curve.png"

    def __post_init__(self) -> None:
        """填充默认环境参数。"""
        if self.env_kwargs is None:
            self.env_kwargs = {}
