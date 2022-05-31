from typing import Any
from ding.framework.storage import Storage
from ding.utils import read_file, save_file


class FileStorage(Storage):

    def save(self, data: Any) -> None:
        save_file(self.path, data)

    def load(self) -> Any:
        return read_file(self.path)
