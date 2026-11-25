from dataclasses import dataclass, field
from typing import Optional, Tuple

VERSIONS = ("v1", "v2", "v3")
TASKS = ("CartPole-v1", "LunarLander-v3")
SOLVED_RETURN = {"CartPole-v1": 475.0, "LunarLander-v3": 200.0}


@dataclass(frozen=True)
class DreamerConfig:
    """低维向量观测、离散动作的 Dreamer 网络与训练参数。"""

    obs_dim: int
    action_dim: int
    embed_dim: int = 64
    deter_dim: int = 128
    stoch_dim: int = 16
    stoch_classes: int = 16
    hidden_dim: int = 128
    # 想象轨迹长度与折扣时间尺度是独立的参数。
    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95

    model_lr: float = 3e-4
    actor_lr: Optional[float] = None
    critic_lr: float = 3e-4
    grad_clip: float = 100.0
    entropy_scale: Optional[float] = None
    free_nats: Optional[float] = None
    kl_scale: float = 1.0
    discount_scale: float = 1.0
    kl_balance: float = 0.8
    target_interval: int = 100  # V2 周期性复制 critic 参数
    target_tau: float = 0.02  # V3 用慢速 critic 做正则
    dynamics_scale: float = 1.0  # V3 Nature 版
    representation_scale: float = 0.1
    replay_value_scale: float = 0.3
    slow_value_scale: float = 1.0
    return_ema: float = 0.01
    unimix: float = 0.01
    bins: int = 255


@dataclass
class TrainConfig:
    """采样、更新频率、评估协议和实验输出。"""

    env_ids: Tuple[str, ...] = TASKS
    agent_versions: Tuple[str, ...] = VERSIONS
    seeds: Tuple[int, ...] = (0, 1, 2)
    # 未指定时取环境对应的预算和评估频率。
    total_steps: Optional[int] = None
    seed_steps: Optional[int] = None
    eval_every: Optional[int] = None
    pretrain_steps: int = 100
    train_every: int = 4
    train_steps: int = 1
    batch_size: int = 16
    seq_len: int = 16
    replay_size: int = 200_000
    log_every: int = 1000
    eval_episodes: int = 20
    holdout_episodes: int = 100
    eval_workers: int = 16
    workdir: str = "runs/dreamer"
    device: str = "cpu"
    threads: int = 1
    use_obs_norm: bool = True
    stop_when_solved: bool = True
    plot: bool = True
    env_kwargs: dict = field(default_factory=dict)
