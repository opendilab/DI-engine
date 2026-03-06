from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

class ReplayBuffer:
    """按 Episode 存储，支持采样固定长度的时间序列片段。"""
    def __init__(self, capacity_steps: int):
        self.capacity = capacity_steps
        self.episodes = [] # 存储完整轨迹的列表
        self.total_steps = 0

    def __len__(self) -> int:
        """返回回放中累计的环境步数。"""
        return self.total_steps

    def add_episode(self, obs, actions, rewards, dones):
        """存入一条完整的 Episode 轨迹。"""
        if not (len(obs) == len(actions) == len(rewards) == len(dones)):
            raise ValueError("Episode fields must have the same length.")
        episode = {
            "obs": np.array(obs, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }
        self.episodes.append(episode)
        self.total_steps += len(obs)
        # 移除旧数据以维持容量限制
        while self.total_steps > self.capacity:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["obs"])

    def sample(self, batch_size: int, seq_len: int):
        """核心功能：采样 [B, T, ...] 格式的序列片段。"""
        # 1. 筛选出长度足够的轨迹
        candidates = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
        if not candidates:
            raise ValueError("Not enough data to sample sequences.")
        
        batch = {"obs": [], "actions": [], "rewards": [], "dones": []}
        for _ in range(batch_size):
            ep = random.choice(candidates)
            # 2. 在轨迹中随机选择起始点
            start = random.randint(0, len(ep["obs"]) - seq_len)
            end = start + seq_len
            
            # 3. 切片提取
            for k in batch:
                batch[k].append(ep[k][start:end])
        
        # 4. 堆叠为 Tensor: [Batch, Time, Dim]
        return {k: np.stack(v) for k, v in batch.items()}
