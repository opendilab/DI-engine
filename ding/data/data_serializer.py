from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
from time import sleep, time
from ditk import logging
import sys
import numpy as np
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union
from ding.data import FileStorage, Storage
from os import path
import uuid
import pickle
from ding.data.shm_buffer import ShmBuffer
from ding.framework.supervisor import RecvPayload, Supervisor, ChildType, SendPayload, SharedObject


class StorageWorker:

    def load(self, storage: Storage) -> Any:
        return storage.load()


class StorageLoader(Supervisor):
    """
    Overview:
        Load data storage in shadow processes.
    """

    def __init__(self) -> None:
        super().__init__(type_=ChildType.PROCESS)
        self._load_lock = Lock()  # Load (first meet) should be called one by one.
        self._load_queue = queue.Queue()  # Queue to be sent to child processes.
        self._callback_map: Dict[str, Callable] = {}
        self._shm_obj_map: Dict[int, SharedObject] = {}
        self._idle_proc_ids = set()
        self._child_num = 3

    def shutdown(self, timeout: Optional[float] = None) -> None:
        super().shutdown(timeout)
        self._recv_loop = None

    def start_link(self) -> None:
        super().start_link()
        self._recv_loop = Thread(target=self._loop_recv, daemon=True)
        self._recv_loop.start()
        self._send_loop = Thread(target=self._loop_send, daemon=True)
        self._send_loop.start()

    def load(self, storage: Storage, callback: Callable):
        with self._load_lock:
            if not self._running:
                self._first_meet(storage, callback)
                return
        self._load_queue.put([storage, callback])

    def _first_meet(self, storage: Storage, callback: Callable):
        """
        Overview:
            When first meet an object type, we'll load this object directly and analysis the structure,
            to allocate the shared memory object and create subprocess workers.
        """
        obj = storage.load()
        # Create three workers for each usage type.
        for i in range(self._child_num):
            shm_obj = self._create_shared_object(obj)
            self._shm_obj_map[i] = shm_obj
            self.register(lambda: StorageWorker(), shared_object=shm_obj)
        self._idle_proc_ids = set(range(self._child_num))
        self.start_link()
        callback(obj)

    def _loop_recv(self):
        while True:
            payload = self.recv(ignore_err=True)
            if payload.err:
                logging.warning("Got error when loading data: {}".format(payload.err))
                if payload.req_id in self._callback_map:
                    del self._callback_map[payload.req_id]
            else:
                self._shm_putback(payload, self._shm_obj_map[payload.proc_id].buf)
                if payload.req_id in self._callback_map:
                    callback = self._callback_map.pop(payload.req_id)
                    callback(payload.data)
            self._idle_proc_ids.add(payload.proc_id)

    def _loop_send(self):
        while True:
            storage, callback = self._load_queue.get()
            while not self._idle_proc_ids:
                sleep(0.01)
            proc_id = self._idle_proc_ids.pop()
            payload = SendPayload(proc_id=proc_id, method="load", args=[storage])
            self._callback_map[payload.req_id] = callback
            self.send(payload)

    def _create_shared_object(self, obj: Union[Dict, List]) -> SharedObject:
        """
        Overview:
            Create shared object (buf and callback) by walk through the data structure.
        """

        def to_shm(obj: Dict):
            shm_buf = {}
            for key, val in obj.items():
                if isinstance(val, np.ndarray):
                    shm_buf[key] = ShmBuffer(val.dtype, val.shape, copy_on_get=False)
            return shm_buf

        if isinstance(obj, Dict):
            shm_buf = to_shm(obj)
        elif isinstance(obj, List):
            shm_buf = [to_shm(o) for o in obj]
        else:
            raise ValueError("Invalid obj type ({})".format(type(obj)))
        return SharedObject(buf=shm_buf, callback=self._shm_callback)

    def _shm_callback(self, payload: RecvPayload, buf: Union[Dict, List]):
        """
        Overview:
            Called in subprocess, put payload.data into buf.
        """
        assert type(
            payload.data
        ) is type(buf), "Data type ({}) and buf type ({}) are not match!".format(type(payload.data), type(buf))
        if isinstance(buf, Dict):
            for key, val in buf.items():
                val.fill(payload.data[key])
                payload.data[key] = None
        elif isinstance(buf, List):
            for i, buf_ in enumerate(buf):
                for key, val in buf_.items():
                    val.fill(payload.data[i][key])
                    payload.data[i][key] = None

    def _shm_putback(self, payload: RecvPayload, buf: Union[Dict, List]):
        """
        Overview:
            Called in main process, put buf back into payload.data.
        """
        assert type(
            payload.data
        ) is type(buf), "Data type ({}) and buf type ({}) are not match!".format(type(payload.data), type(buf))
        if isinstance(buf, Dict):
            for key, val in buf.items():
                payload.data[key] = val.get()
        elif isinstance(buf, List):
            for i, buf_ in enumerate(buf):
                for key, val in buf_.items():
                    payload.data[i][key] = val.get()


@dataclass
class StorageObject:
    storage: Storage
    usage_type: str


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
        self._storage_loaders = {}

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
        if self._storage_loaders:
            for loader in self._storage_loaders.values():
                loader.shutdown()
            self._storage_loaders = None
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
        self._from_storage(obj, callback)

    def _to_storage(self, obj: Union[Dict, List]) -> Union[Dict, List, Storage]:
        """
        Overview:
            Convert large data (>10MB) into data storage object.
        """
        size = self._get_size(obj)
        if size > 10485760:
            if isinstance(obj, Dict):
                usage_type = ",".join(obj.keys())[:20]
            elif isinstance(obj, List) and isinstance(obj[0], Dict):
                usage_type = ",".join(obj[0].keys())[:20]
            else:
                logging.warning(
                    "Object size ({}) is too large but the object type ({}) is not supported (Dict or List) in serializer."
                    .format(size, type(obj))
                )
                return obj
            return StorageObject(storage=self._storage_fn(obj), usage_type=usage_type)
        else:
            return obj

    def _from_storage(self, obj: Union[Dict, List, StorageObject], callback: Callable):
        """
        Overview:
            Convert data storage object to real data.
        """
        if isinstance(obj, StorageObject):
            if obj.usage_type not in self._storage_loaders:
                self._storage_loaders[obj.usage_type] = StorageLoader()
            self._storage_loaders[obj.usage_type].load(obj.storage, callback)
        else:
            callback(obj)

    def _get_size(self, obj: Union[Dict, List]) -> int:
        """
        Overview:
            This method will only check ndarray that may generate
            a large memory footprint and return the sum of them.
            Note that only the topmost object of dict will be checked, not each child object recursively.
        """
        size = 0
        if isinstance(obj, Dict):
            for val in obj.values():
                if isinstance(val, np.ndarray):
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
