from typing import List
from nervex.worker.replay_buffer import NaiveReplayBuffer
from nervex.utils import BUFFER_REGISTRY


@BUFFER_REGISTRY.register('episode')
class EpisodeReplayBuffer(NaiveReplayBuffer):

    @property
    def episode_len(self) -> List[int]:
        return [len(episode) for episode in self._data]
