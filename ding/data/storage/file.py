from typing import Any
from ding.data.storage import Storage
import pickle

from ding.utils.file_helper import read_file, save_file


class FileStorage(Storage):

    def save(self, data: Any) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> Any:
        with open(self.path, "rb") as f:
            return pickle.load(f)


class FileModelStorage(Storage):

    def save(self, state_dict: object) -> None:
        save_file(self.path, state_dict)

    def load(self) -> object:
        return read_file(self.path)
