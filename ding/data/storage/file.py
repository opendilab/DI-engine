from typing import Any
from ding.data.storage import Storage
import pickle


class FileStorage(Storage):

    def save(self, data: Any) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> Any:
        with open(self.path, "rb") as f:
            return pickle.load(f)
