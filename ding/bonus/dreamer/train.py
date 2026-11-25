import argparse
import csv
import json
import os
import random
import time
from contextlib import ExitStack, closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym

if __package__:
    from .config import TASKS, VERSIONS, SOLVED_RETURN, DreamerConfig, TrainConfig
    from .dreamer.agent import make_agent
    from .dreamer.replay import ReplayBuffer
else:
    from config import TASKS, VERSIONS, SOLVED_RETURN, DreamerConfig, TrainConfig
    from dreamer.agent import make_agent
    from dreamer.replay import ReplayBuffer

# 预算与评估频率按策略决策次数计。输出另记录原始环境步数。
TASK_BUDGETS = {"CartPole-v1": (60_000, 1000, 2000), "LunarLander-v3": (220_000, 4000, 5000)}


class ObsNormalizer:
    """V1/V2 可用随机预采样的固定统计量归一化向量观测。"""

    def __init__(self, observations: np.ndarray):
        self.mean = observations.mean(0, dtype=np.float64).astype(np.float32)
        self.std = observations.std(0, dtype=np.float64).clip(min=1e-3).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / self.std, -10.0, 10.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_env(env, seed=None):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action, repeat: int = 1):
    if repeat < 1:
        raise ValueError("action_repeat must be positive.")
    total_reward = 0.0
    for elapsed in range(1, repeat + 1):
        out = env.step(action)
        if len(out) == 5:
            obs, reward, terminal, truncated, info = out
            done = terminal or truncated
        else:
            obs, reward, done, info = out
            terminal = done and not info.get("TimeLimit.truncated", False)
        total_reward += reward
        if done:
            break
    return obs, total_reward, done, terminal, {**info, "repeat_steps": elapsed}


def evaluate(
        agent,
        env_id: str,
        episodes: int,
        seed: int,
        normalizer: Optional[ObsNormalizer] = None,
        env_kwargs=None,
        workers: int = 16,
        action_repeat: int = 1,
) -> np.ndarray:
    """对固定种子的真实环境做贪心动作评估，并保留每局的原始回报。"""
    cuda_devices = []
    if agent.device.type == "cuda":
        index = agent.device.index if agent.device.index is not None else torch.cuda.current_device()
        cuda_devices.append(index)
    returns = np.zeros(episodes, dtype=np.float64)
    # RSSM 后验依然采样。隔离评估的 Torch RNG，避免评估频率改变训练序列。
    with torch.random.fork_rng(devices=cuda_devices), ExitStack() as stack:
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        envs = [
            stack.enter_context(closing(gym.make(env_id, **(env_kwargs or {})))) for _ in range(min(workers, episodes))
        ]
        for first in range(0, episodes, len(envs)):
            group = envs[:episodes - first]
            batch_size = len(group)
            obs = np.stack([reset_env(env, seed + first + i) for i, env in enumerate(group)])
            state = agent.init_state(batch_size)
            prev_action = torch.zeros(batch_size, agent.config.action_dim, device=agent.device)
            done = np.zeros(batch_size, dtype=bool)
            while not done.all():
                state = agent.observe(normalizer(obs) if normalizer else obs, state, prev_action)
                actions = agent.policy_actions(state, eval_mode=True)
                prev_action = F.one_hot(actions, agent.config.action_dim).float()
                actions = actions.cpu().numpy()
                for i in np.flatnonzero(~done):
                    obs[i], reward, done[i], _, _ = step_env(group[i], int(actions[i]), action_repeat)
                    returns[first + i] += reward
    return returns


def _save_checkpoint(
        path: Path, agent, normalizer: Optional[ObsNormalizer], step: int, env_steps: int, action_repeat: int
) -> None:
    normalizer_state = None
    if normalizer is not None:
        normalizer_state = {"mean": torch.from_numpy(normalizer.mean), "std": torch.from_numpy(normalizer.std)}
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "version": agent.version,
            "agent_config": asdict(agent.config),
            "env_steps": env_steps,
            "decision_steps": step,
            "action_repeat": action_repeat,
            "world_model": agent.world_model.state_dict(),
            "actor": agent.actor.state_dict(),
            "normalizer": normalizer_state,
        }, tmp
    )
    tmp.replace(path)


def train(cfg: TrainConfig, env_id: str, version: str, seed: int) -> dict:
    """训练一个环境、版本、种子的组合，写出逐次评估和最终独立评估。"""
    torch.set_num_threads(cfg.threads)
    seed_everything(seed)
    version = version.lower()
    lunar = env_id == "LunarLander-v3"
    lunar_v2 = lunar and version == "v2"
    if cfg.action_repeat is None:
        repeat = 3 if lunar and version in ("v1", "v2") else 1
        cfg = replace(cfg, action_repeat=repeat)
    total_steps, seed_steps, eval_every = (
        value if value is not None else default for value, default in
        zip((cfg.total_steps, cfg.seed_steps, cfg.eval_every), TASK_BUDGETS.get(env_id, TASK_BUDGETS["CartPole-v1"]))
    )
    if min(total_steps, eval_every, cfg.train_every, cfg.train_steps, cfg.batch_size, cfg.eval_episodes,
           cfg.holdout_episodes, cfg.eval_workers, cfg.log_every, cfg.action_repeat) <= 0 or cfg.seq_len < 2:
        raise ValueError("Step counts and batch sizes must be positive, and seq_len must be at least 2.")
    if min(seed_steps, cfg.pretrain_steps) < 0:
        raise ValueError("seed_steps and pretrain_steps must be non-negative.")

    with closing(gym.make(env_id, **cfg.env_kwargs)) as env:
        if not isinstance(env.action_space, gym.spaces.Discrete) or getattr(env.action_space, "start", 0) != 0:
            raise ValueError(f"{env_id} must use a zero-based Discrete action space.")
        if len(env.observation_space.shape) != 1:
            raise ValueError(f"{env_id} must provide vector observations.")
        actor_lr = cfg.actor_lr if cfg.actor_lr is not None else (8e-5 if lunar_v2 else None)
        dynamics_mix = cfg.v2_dynamics_mix if cfg.v2_dynamics_mix is not None else (0.9 if lunar_v2 else 0.0)
        agent_config = DreamerConfig(
            obs_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.n),
            actor_lr=actor_lr,
            # V2 的离散 Atari 配置也采用较低的 KL 权重。
            kl_scale=0.1 if env_id == "CartPole-v1" and version == "v2" else 1.0,
            v2_dynamics_mix=dynamics_mix,
        )
        agent = make_agent(version, agent_config, torch.device(cfg.device))
        env.action_space.seed(seed)
        return _train_loop(cfg, env, env_id, agent, seed, total_steps, seed_steps, eval_every)


def _train_loop(
        cfg: TrainConfig, env, env_id: str, agent, seed: int, total_steps: int, seed_steps: int, eval_every: int
) -> dict:
    seed_dir = Path(cfg.workdir) / env_id / agent.version / f"seed_{seed}"
    metrics_dir = seed_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    run_config = {
        "env_id": env_id,
        "version": agent.version,
        "seed": seed,
        "gym_version": gym.__version__,
        "torch_version": str(torch.__version__),
        "train": {
            **asdict(cfg), "total_steps": total_steps,
            "seed_steps": seed_steps,
            "eval_every": eval_every
        },
        "agent": asdict(agent.config),
    }
    (seed_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"{env_id} {agent.version} seed={seed} device={agent.device} workdir={seed_dir}", flush=True)

    replay = ReplayBuffer(cfg.replay_size, seed=seed)
    normalizer = None
    norm_observations = []
    obs = reset_env(env, seed)
    state = agent.init_state(1)
    prev_action = torch.zeros(1, agent.config.action_dim, device=agent.device)
    episode = ([obs.copy()], [-1], [0.], [False])
    action_ids = torch.arange(agent.config.action_dim, device=agent.device)
    pretrain = cfg.pretrain_steps
    updates = 0
    env_steps = 0
    stable = 0
    holdouts = []
    threshold = SOLVED_RETURN.get(env_id)
    started = time.monotonic()
    metrics_path = metrics_dir / "eval_returns.csv"
    with metrics_path.open("w", newline="") as out:
        csv.writer(out).writerow(
            ["env_steps", "decision_steps", "avg_return", "seed", "std_return", "episodes", "updates", "elapsed_s"]
        )

    def evaluate_policy(step: int) -> float:
        scores = evaluate(
            agent, env_id, cfg.eval_episodes, 100_000, normalizer, cfg.env_kwargs, cfg.eval_workers, cfg.action_repeat
        )
        with metrics_path.open("a", newline="") as out:
            csv.writer(out).writerow(
                [env_steps, step,
                 scores.mean(), seed,
                 scores.std(),
                 len(scores), updates,
                 time.monotonic() - started]
            )
        print(
            f"eval step={step} env_steps={env_steps} updates={updates} "
            f"return={scores.mean():.2f} std={scores.std():.2f}",
            flush=True
        )
        return float(scores.mean())

    def confirm(step: int) -> dict:
        # 每次确认使用一批新初始种子。与训练和用于触发早停的评估种子分开。
        eval_seed = 200_000 + len(holdouts) * cfg.holdout_episodes
        scores = evaluate(
            agent, env_id, cfg.holdout_episodes, eval_seed, normalizer, cfg.env_kwargs, cfg.eval_workers,
            cfg.action_repeat
        )
        result = {
            "step": step,
            "env_steps": env_steps,
            "seed_start": eval_seed,
            "episodes": len(scores),
            "threshold": threshold,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "returns": scores.tolist(),
            "solved": bool(threshold is not None and stable >= 2 and scores.mean() >= threshold),
        }
        holdouts.append(result)
        print(f"holdout step={step} return={result['mean']:.2f} solved={result['solved']}", flush=True)
        return result

    def sample_batch() -> dict:
        batch = replay.sample(cfg.batch_size, cfg.seq_len)
        if normalizer is not None:
            batch["obs"] = normalizer(batch["obs"])
        return batch

    evaluate_policy(0)
    for step in range(1, total_steps + 1):
        if step <= seed_steps:
            action = int(env.action_space.sample())
            if cfg.use_obs_norm and agent.version != "v3":
                norm_observations.append(obs.copy())
        else:
            state = agent.observe(normalizer(obs) if normalizer else obs, state, prev_action)
            action = agent.policy(state)
        # 动作重复以环境返回的奖励求和，回合结束立即停止重复。指标另计原始环境步数。
        obs, reward, done, terminated, info = step_env(env, action, cfg.action_repeat)
        env_steps += info["repeat_steps"]
        for column, value in zip(episode, (obs.copy(), action, reward, terminated)):
            column.append(value)
        if done:
            # 截断也结束采集回合并清空 RSSM，但 continuation 只看真实终止。
            replay.add_episode(*episode)
            obs = reset_env(env)
            episode = ([obs.copy()], [-1], [0.], [False])
            state = agent.init_state(1)
            prev_action.zero_()
        else:
            prev_action = (action_ids == action).float().unsqueeze(0)

        if step == seed_steps:
            if norm_observations:
                normalizer = ObsNormalizer(np.stack(norm_observations))
                norm_observations.clear()
            # 可能正在一个预采样回合中。此时续接刚刚执行的动作，从零状态推断一次后验。
            state = agent.init_state(1)

        if step >= seed_steps and replay.can_sample(cfg.seq_len):
            while pretrain:
                agent.train_step(sample_batch(), behavior=False, report=False)
                pretrain -= 1
            if step % cfg.train_every == 0:
                for update in range(cfg.train_steps):
                    report = step % cfg.log_every == 0 and update == cfg.train_steps - 1
                    metrics = agent.train_step(sample_batch(), report=report)
                    updates += 1
                    if metrics is not None:
                        if not np.isfinite(list(asdict(metrics).values())).all():
                            raise FloatingPointError(f"Non-finite loss at decision step {step}: {metrics}")
                        print(
                            f"train step={step} updates={updates} model={metrics.model_loss:.3f} "
                            f"actor={metrics.actor_loss:.3f} critic={metrics.critic_loss:.3f}",
                            flush=True
                        )

        if step % eval_every == 0 or step == total_steps:
            score = evaluate_policy(step)
            stable = stable + 1 if threshold is not None and score >= threshold else 0
            if step == total_steps or (cfg.stop_when_solved and stable >= 2):
                if confirm(step)["solved"]:
                    break

    _save_checkpoint(seed_dir / "policy.pt", agent, normalizer, step, env_steps, cfg.action_repeat)
    result = {
        "env_id": env_id,
        "version": agent.version,
        "seed": seed,
        "env_steps": env_steps,
        "decision_steps": step,
        "updates": updates,
        "elapsed_s": time.monotonic() - started,
        "solved": holdouts[-1]["solved"],
        "holdouts": holdouts,
    }
    tmp_result = seed_dir / "result.tmp"
    tmp_result.write_text(json.dumps(result, indent=2) + "\n")
    tmp_result.replace(seed_dir / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dreamer V1/V2/V3 on vector observations and discrete actions.")
    parser.add_argument("--env", nargs="+", choices=TASKS)
    parser.add_argument("--versions", nargs="+", choices=VERSIONS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", dest="total_steps", type=int)
    parser.add_argument("--seed-steps", type=int)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--v2-dynamics-mix", type=float, help="V2 pathwise fraction (1 - rho in the paper).")
    parser.add_argument("--pretrain-steps", type=int)
    parser.add_argument("--train-every", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--holdout-episodes", type=int)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--workdir")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=None)
    parser.add_argument("--full-budget", dest="stop_when_solved", action="store_false", default=None)
    args = vars(parser.parse_args())
    args["env_ids"] = args.pop("env")
    args["agent_versions"] = args.pop("versions")
    cfg = TrainConfig(**{key: value for key, value in args.items() if value is not None})
    if args["workdir"] is None and Path(cfg.workdir).exists():
        cfg.workdir = f"{cfg.workdir}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    for env_id in cfg.env_ids:
        for version in cfg.agent_versions:
            for seed in cfg.seeds:
                train(cfg, env_id, version, seed)
    if cfg.plot:
        if __package__:
            from .plot_returns import plot_returns
        else:
            from plot_returns import plot_returns
        plot_returns(workdir=cfg.workdir, agent_versions=cfg.agent_versions)


if __name__ == "__main__":
    main()
