from typing import List
from ding.worker.replay_buffer import NaiveReplayBuffer
from ding.utils import BUFFER_REGISTRY


@BUFFER_REGISTRY.register('episode')
class EpisodeReplayBuffer(NaiveReplayBuffer):
    r"""
    Overview:
        Episode replay buffer is a buffer to store complete episodes, i.e. Each element in episode buffer is an episode.
        Some algorithms do not want to sample `batch_size` complete episodes, however, they want some transitions with
        some fixed length. As a result, ``sample`` should be overwritten for those requirements.
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    """

    @property
    def episode_len(self) -> List[int]:
        return [len(episode) for episode in self._data]
