from queue import LifoQueue
from collections import deque


class LifoDeque(LifoQueue):
    """
    Like LifoQueue, but automatically replaces the oldest data when the queue is full.
    """

    def _init(self, maxsize):
        self.maxsize = maxsize + 1
        self.queue = deque(maxlen=maxsize)
