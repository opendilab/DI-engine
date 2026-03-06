import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import TrainConfig


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取评估 CSV 文件。"""
    steps: List[int] = []
    returns: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["env_steps"]))
            returns.append(float(row["avg_return"]))
    return np.asarray(steps, dtype=np.float32), np.asarray(returns, dtype=np.float32)


def collect_runs(workdir: str) -> Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]]:
    """收集所有实验的评估曲线。"""
    runs: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for root, _, files in os.walk(workdir):
        if "eval_returns.csv" not in files:
            continue
        csv_path = os.path.join(root, "eval_returns.csv")
        rel = os.path.relpath(root, workdir)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        env_id, version = parts[0], parts[1]
        steps, returns = load_csv(csv_path)
        if len(steps) == 0:
            continue
        runs[(env_id, version)].append((steps, returns))
    return runs


def aggregate_group(series: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对同一组实验按步数聚合均值与方差。"""
    step_to_values: Dict[int, List[float]] = defaultdict(list)
    for steps, returns in series:
        for s, r in zip(steps, returns):
            step_to_values[int(s)].append(float(r))
    all_steps = np.array(sorted(step_to_values.keys()), dtype=np.float32)
    means = []
    stds = []
    for s in all_steps:
        vals = np.asarray(step_to_values[int(s)], dtype=np.float32)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return all_steps, np.asarray(means), np.asarray(stds)


def env_plot_path(base_path: str, env_id: str) -> str:
    """根据环境名生成单独图片路径。"""
    root, ext = os.path.splitext(base_path)
    safe_env = env_id.replace("/", "_")
    if not ext:
        ext = ".png"
    return f"{root}_{safe_env}{ext}"


def plot_returns(
    workdir: Optional[str] = None,
    plot_path: Optional[str] = None,
    agent_versions: Optional[Sequence[str]] = None,
) -> None:
    """按环境分别绘制学习曲线。"""
    cfg = TrainConfig()
    resolved_workdir = workdir if workdir is not None else cfg.workdir
    resolved_plot_path = plot_path if plot_path is not None else cfg.plot_path
    resolved_versions = list(agent_versions) if agent_versions is not None else list(cfg.agent_versions)

    runs = collect_runs(resolved_workdir)
    if not runs:
        print("No metrics found. Run training first.")
        return

    env_ids = sorted({env_id for env_id, _ in runs.keys()})
    plot_dir = os.path.dirname(resolved_plot_path) or "."
    os.makedirs(plot_dir, exist_ok=True)
    for env_id in env_ids:
        plt.figure(figsize=(10, 6))
        found = False
        for version in resolved_versions:
            series = runs.get((env_id, version), [])
            if not series:
                continue
            steps, mean, std = aggregate_group(series)
            if len(steps) == 0:
                continue
            found = True
            x = steps / 1000.0
            label = version.upper()
            plt.plot(x, mean, label=label)
            if len(series) > 1:
                plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        if not found:
            plt.close()
            continue
        plt.xlabel("Env Steps (k)")
        plt.ylabel("Return")
        plt.title(f"Dreamer V1/V2/V3 on {env_id}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = env_plot_path(resolved_plot_path, env_id)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    plot_returns()
