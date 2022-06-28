from abc import ABC, abstractmethod
import os
import queue
from time import sleep, time
from ditk import logging
import numpy as np
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union

from ding.data import FileStorage, Storage
from os import path
import uuid
from ding.data.shm_buffer import ShmBuffer
from ding.framework.supervisor import RecvPayload, Supervisor, ChildType, SendPayload, SharedObject


class StorageWorker:

    def load(self, storage: Storage) -> Any:
        return storage.load()


class StorageLoader(Supervisor, ABC):
    """
    Overview:
        Load data storage in shadow processes.
    """

    def __init__(self, worker_num: int = 3) -> None:
        super().__init__(type_=ChildType.PROCESS)
        self._load_lock = Lock()  # Load (first meet) should be called one by one.
        self._load_queue = queue.Queue()  # Queue to be sent to child processes.
        self._callback_map: Dict[str, Callable] = {}
        self._shm_obj_map: Dict[int, SharedObject] = {}
        self._idle_proc_ids = set()
        self._worker_num = worker_num

    def shutdown(self, timeout: Optional[float] = None) -> None:
        super().shutdown(timeout)
        self._recv_loop = None
        self._send_loop = None

    def start_link(self) -> None:
        if not self._running:
            super().start_link()
            self._recv_loop = Thread(target=self._loop_recv, daemon=True)
            self._recv_loop.start()
            self._send_loop = Thread(target=self._loop_send, daemon=True)
            self._send_loop.start()

    @abstractmethod
    def to_storage(self, obj: Union[Dict, List]) -> Storage:
        raise NotImplementedError

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
        for i in range(self._worker_num):
            shm_obj = self._create_shared_object(obj)
            self._shm_obj_map[i] = shm_obj
            self.register(StorageWorker, shared_object=shm_obj)
        self._idle_proc_ids = set(range(self._worker_num))
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


class FileStorageLoader(StorageLoader):

    def __init__(self, dirname: str, ttl: int = 600, worker_num: int = 3) -> None:
        """
        Overview:
            Dump and load object with file storage.
        Arguments:
            - dirname (:obj:`str`): The directory to save files.
            - ttl (:obj:`str`): Maximum time to keep a file, after which it will be deleted.
            - worker_num (:obj:`int`): Number of subprocess worker loaders.
        """
        super().__init__(worker_num)
        self._dirname = dirname
        self._files = []
        self._cleanup_thread = None
        self._ttl = ttl  # # Delete files created 10 minutes ago.

    def to_storage(self, obj: Union[Dict, List]) -> FileStorage:
        if not path.exists(self._dirname):
            os.mkdir(self._dirname)
        filename = "{}.pkl".format(uuid.uuid1())
        full_path = path.join(self._dirname, filename)
        f = FileStorage(full_path)
        f.save(obj)
        self._files.append([time(), f.path])
        self._start_cleanup()
        return f

    def _start_cleanup(self):
        """
        Overview:
            Start a cleanup thread to clean up files that are taking up too much time on the disk.
        """
        if self._cleanup_thread is None:
            self._cleanup_thread = Thread(target=self._loop_cleanup, daemon=True)
            self._cleanup_thread.start()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        super().shutdown(timeout)
        self._cleanup_thread = None

    def _loop_cleanup(self):
        while True:
            if len(self._files) == 0 or time() - self._files[0][0] < self._ttl:
                sleep(1)
                continue
            _, file_path = self._files.pop(0)
            if path.exists(file_path):
                os.remove(file_path)
