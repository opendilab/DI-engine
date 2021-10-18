from typing import Any, List
from ding.worker.buffer import Buffer
from ding.worker.buffer.storage import Storage


class NaiveBuffer(Buffer):

    def __init__(self, storage: Storage, **kwargs) -> None:
        super().__init__(storage, **kwargs)

    def push(self, data: Any) -> None:
        self.storage.append(data)

    def sample(self, size: int) -> List[Any]:
        return self.storage.sample(size)

    def clear(self) -> None:
        self.storage.clear()
