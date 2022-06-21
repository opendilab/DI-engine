from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Callable, Dict, List, Union
import pickle


class DataSerializer:
    """
    A data serializer that will encode/decode data in subprocesses.
    """

    def __init__(self) -> None:
        self._serializer_pool = None
        self._running = False

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
        s = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        callback(s)

    def _load(self, s: bytes, callback: Callable):
        obj = pickle.loads(s)
        callback(obj)

    def __del__(self):
        self.stop()
