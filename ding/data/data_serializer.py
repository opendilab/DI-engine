from concurrent.futures import ThreadPoolExecutor
import sys
from typing import Callable, Dict, List, Optional, Union
from ding.data import FileStorage, Storage
from os import path
import uuid
import pickle
import torch
import numpy as np


class DataSerializer:
    """
    A data serializer that will encode/decode data in subprocesses.
    """

    def __init__(self, storage_type: type = FileStorage, dirname: Optional[str] = None) -> None:
        self._serializer_pool = None
        self._running = False
        self._storage_type = storage_type
        self._dirname = dirname
        self._storage_fn = self._get_storage_fn()

    def _get_storage_fn(self) -> Callable:
        if self._storage_type is FileStorage:
            assert path.isdir(self._dirname), "Path {} is not a valid directory!".format(self._dirname)
            return self._to_file_storage
        else:
            raise RuntimeError("Invalid storage type: {}".format(self._storage_type))

    def start(self):
        self._serializer_pool = ThreadPoolExecutor(max_workers=2)
        self._running = True
        return self

    def stop(self):
        if not self._running:
            return
        if self._serializer_pool is not None:
            self._serializer_pool.shutdown()
            self._serializer_pool = None
        self._running = False

    def dump(self, obj: Union[Dict, List], callback: Callable):
        assert self._running, "Data serializer is not running, call start first."
        self._serializer_pool.submit(self._dump, obj, callback)

    def load(self, s: bytes, callback: Callable):
        assert self._running, "Data serializer is not running, call start first."
        self._serializer_pool.submit(self._load, s, callback)

    def _dump(self, obj: Union[Dict, List], callback: Callable):
        obj = self._to_storage(obj)
        s = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        callback(s)

    def _load(self, s: bytes, callback: Callable):
        obj = pickle.loads(s)
        obj = self._from_storage(obj)
        callback(obj)

    def _to_storage(self, obj: Union[Dict, List]) -> Union[Dict, List, Storage]:
        """
        Overview:
            Convert large data (>10MB) into data storage object.
        """
        if self._get_size(obj) > 10485760:
            return self._storage_fn(obj)
        else:
            return obj

    def _from_storage(self, obj: Union[Dict, List, Storage]) -> Union[Dict, List]:
        """
        Overview:
            Convert data storage object to real data.
        """
        if isinstance(obj, Storage):
            return obj.load()
        else:
            return obj

    def _get_size(self, obj: Union[Dict, List]) -> int:
        """
        Overview:
            This method will check several structure types that may generate
            a large memory footprint (tensor or ndarray) and return the sum of them.
            Note that only the topmost object of dict will be checked, not each child object recursively.
        """
        size = 0
        if isinstance(obj, Dict):
            for val in obj.values():
                if isinstance(val, torch.Tensor):
                    size += sys.getsizeof(val.storage())
                else:
                    size += sys.getsizeof(val)
            return size
        elif isinstance(obj, List):
            return sum([self._get_size(o) for o in obj])
        else:
            return sys.getsizeof(obj)

    def _to_file_storage(self, obj: Union[Dict, List]) -> Storage:
        filename = "{}.pkl".format(uuid.uuid1())
        full_path = path.join(self._dirname, filename)
        f = FileStorage(full_path)
        f.save(obj)
        return f

    def __del__(self):
        self.stop()
