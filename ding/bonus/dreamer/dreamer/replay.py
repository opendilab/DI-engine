from collections import deque

import numpy as np


class ReplayBuffer:
    """按完整回合存储，均匀采样连续窗口，时序批次形状为 [B,T,...]。"""

    def __init__(self, capacity_steps: int, seed=None):
        if capacity_steps <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = capacity_steps
        self.episodes = deque()
        self.total_steps = 0
        self.rng = np.random.default_rng(seed)
        self._index = None

    def __len__(self) -> int:
        return self.total_steps

    def add_episode(self, obs, actions, rewards, dones) -> None:
        """每行存 (x_t, a_{t-1}, r_t, is_terminal_t)，回合首行用 -1 表示零动作。"""
        if not len(obs) or not (len(obs) == len(actions) == len(rewards) == len(dones)):
            raise ValueError("Replay episode fields must be nonempty and have equal lengths.")
        if len(obs) > self.capacity:
            raise ValueError("Replay capacity is smaller than one episode.")
        episode = dict(
            obs=np.asarray(obs, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
        self.episodes.append(episode)
        self.total_steps += len(obs)
        while self.total_steps > self.capacity:
            self.total_steps -= len(self.episodes.popleft()["obs"])
        self._index = None

    def _windows(self, seq_len: int):
        if self._index is None or self._index[0] != seq_len:
            episodes = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
            # 按窗口数而非回合数抽样。每个窗口等概率，避免短回合被过度采样。
            offsets = np.cumsum([0] + [len(ep["obs"]) - seq_len + 1 for ep in episodes])
            self._index = seq_len, episodes, offsets
        return self._index[1:]

    def can_sample(self, seq_len: int) -> bool:
        return self._windows(seq_len)[1][-1] > 0

    def sample(self, batch_size: int, seq_len: int):
        episodes, offsets = self._windows(seq_len)
        if not offsets[-1]:
            raise ValueError("No episode contains a replay window of the requested length.")
        windows = self.rng.integers(offsets[-1], size=batch_size)
        indices = np.searchsorted(offsets[1:], windows, side="right")
        starts = windows - offsets[indices]
        return {
            key: np.stack([episodes[i][key][start:start + seq_len] for i, start in zip(indices, starts)])
            for key in episodes[0]
        }
