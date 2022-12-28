from dataclasses import dataclass
import os
import torch
import numpy as np
import uuid
import treetensor.torch as ttorch
from abc import ABC, abstractmethod
from ditk import logging
from time import sleep, time
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Union
from ding.data import FileStorage, Storage
from os import path
from ding.data.shm_buffer import ShmBuffer
from ding.framework.supervisor import RecvPayload, Supervisor, ChildType, SendPayload


@dataclass
class ShmObject:
    id_: ShmBuffer
    buf: Any


class StorageWorker:

    def load(self, storage: Storage) -> Any:
        return storage.load()


class StorageLoader(Supervisor, ABC):

    def __init__(self, worker_num: int = 3) -> None:
        """
        Overview:
            Save and send data synchronously and load them asynchronously.
        Arguments:
            - worker_num (:obj:`int`): Subprocess worker number.
        """
        super().__init__(type_=ChildType.PROCESS)
        self._load_lock = Lock()  # Load (first meet) should be called one by one.
        self._callback_map: Dict[str, Callable] = {}
        self._shm_obj_map: Dict[int, ShmObject] = {}
        self._worker_num = worker_num
        self._req_count = 0

    def shutdown(self, timeout: Optional[float] = None) -> None:
        super().shutdown(timeout)
        self._recv_loop = None
        self._callback_map = {}
        self._shm_obj_map = {}
        self._req_count = 0

    def start_link(self) -> None:
        if not self._running:
            super().start_link()
            self._recv_loop = Thread(target=self._loop_recv, daemon=True)
            self._recv_loop.start()

    @property
    def _next_proc_id(self):
        return self._req_count % self._worker_num

    @abstractmethod
    def save(self, obj: Union[Dict, List]) -> Storage:
        """
        Overview:
            Save data with a storage object synchronously.
        Arguments:
            - obj (:obj:`Union[Dict, List]`): The data (traj or episodes), can be numpy, tensor or treetensor.
        Returns:
            - storage (:obj:`Storage`): The storage object.
        """
        raise NotImplementedError

    def load(self, storage: Storage, callback: Callable):
        """
        Overview:
            Load data from a storage object asynchronously. \
            This function will analysis the data structure when first meet a new data, \
            then alloc a shared memory buffer for each subprocess, these shared memory buffer \
            will be responsible for asynchronously loading data into memory.
        Arguments:
            - storage (:obj:`Storage`): The storage object.
            - callback (:obj:`Callable`): Callback function after data loaded.
        """
        with self._load_lock:
            if not self._running:
                self._first_meet(storage, callback)
                return

        payload = SendPayload(proc_id=self._next_proc_id, method="load", args=[storage])
        self._callback_map[payload.req_id] = callback
        self.send(payload)
        self._req_count += 1

    def _first_meet(self, storage: Storage, callback: Callable):
        """
        Overview:
            When first meet an object type, we'll load this object directly and analysis the structure,
            to allocate the shared memory object and create subprocess workers.
        Arguments:
            - storage (:obj:`Storage`): The storage object.
            - callback (:obj:`Callable`): Callback function after data loaded.
        """
        obj = storage.load()
        # Create three workers for each usage type.
        for i in range(self._worker_num):
            shm_obj = self._create_shm_buffer(obj)
            self._shm_obj_map[i] = shm_obj
            self.register(StorageWorker, shm_buffer=shm_obj, shm_callback=self._shm_callback)
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
                self._shm_putback(payload, self._shm_obj_map[payload.proc_id])
                if payload.req_id in self._callback_map:
                    callback = self._callback_map.pop(payload.req_id)
                    callback(payload.data)

    def _create_shm_buffer(self, obj: Union[Dict, List]) -> Optional[ShmObject]:
        """
        Overview:
            Create shared object (buf and callback) by walk through the data structure.
        Arguments:
            - obj (:obj:`Union[Dict, List]`): The data (traj or episodes), can be numpy, tensor or treetensor.
        Returns:
            - shm_buf (:obj:`Optional[ShmObject]`): The shared memory buffer.
        """
        max_level = 2

        def to_shm(obj: Dict, level: int):
            if level > max_level:
                return
            shm_buf = None
            if isinstance(obj, Dict) or isinstance(obj, ttorch.Tensor):
                shm_buf = {}
                for key, val in obj.items():
                    # Only numpy array can fill into shm buffer
                    if isinstance(val, np.ndarray):
                        shm_buf[key] = ShmBuffer(val.dtype, val.shape, copy_on_get=False)
                    elif isinstance(val, torch.Tensor):
                        shm_buf[key] = ShmBuffer(
                            val.numpy().dtype, val.numpy().shape, copy_on_get=False, ctype=torch.Tensor
                        )
                    # Recursive parsing structure
                    elif isinstance(val, Dict) or isinstance(val, ttorch.Tensor) or isinstance(val, List):
                        buf = to_shm(val, level=level + 1)
                        if buf:
                            shm_buf[key] = buf
            elif isinstance(obj, List):
                # Double the size of buffer
                shm_buf = [to_shm(o, level=level) for o in obj] * 2
                if all(s is None for s in shm_buf):
                    shm_buf = []
            return shm_buf

        shm_buf = to_shm(obj, level=0)
        if shm_buf is not None:
            random_id = self._random_id()
            shm_buf = ShmObject(id_=ShmBuffer(random_id.dtype, random_id.shape, copy_on_get=False), buf=shm_buf)
        return shm_buf

    def _random_id(self) -> np.ndarray:
        return np.random.randint(1, 9e6, size=(1))

    def _shm_callback(self, payload: RecvPayload, shm_obj: ShmObject):
        """
        Overview:
            Called in subprocess, put payload.data into buf.
        Arguments:
            - payload (:obj:`RecvPayload`): The recv payload with meta info of the data.
            - shm_obj (:obj:`ShmObject`): The shm buffer.
        """
        assert isinstance(payload.data, type(
            shm_obj.buf
        )), "Data type ({}) and buf type ({}) are not match!".format(type(payload.data), type(shm_obj.buf))

        # Sleep while shm object is not ready.
        while shm_obj.id_.get()[0] != 0:
            sleep(0.001)

        max_level = 2

        def shm_callback(data: Union[Dict, List, ttorch.Tensor], buf: Union[Dict, List], level: int):
            if level > max_level:
                return

            if isinstance(buf, List):
                assert isinstance(data, List), "Data ({}) and buf ({}) type not match".format(type(data), type(buf))
            elif isinstance(buf, Dict):
                assert isinstance(data, ttorch.Tensor) or isinstance(
                    data, Dict
                ), "Data ({}) and buf ({}) type not match".format(type(data), type(buf))

            if isinstance(data, Dict) or isinstance(data, ttorch.Tensor):
                for key, val in data.items():
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()
                    buf_val = buf.get(key)
                    if buf_val is None:
                        continue
                    if isinstance(buf_val, ShmBuffer) and isinstance(val, np.ndarray):
                        buf_val.fill(val)
                        data[key] = None
                    else:
                        shm_callback(val, buf_val, level=level + 1)
            elif isinstance(data, List):
                for i, data_ in enumerate(data):
                    shm_callback(data_, buf[i], level=level)

        shm_callback(payload.data, buf=shm_obj.buf, level=0)
        id_ = self._random_id()
        shm_obj.id_.fill(id_)
        payload.extra = id_

    def _shm_putback(self, payload: RecvPayload, shm_obj: ShmObject):
        """
        Overview:
            Called in main process, put buf back into payload.data.
        Arguments:
            - payload (:obj:`RecvPayload`): The recv payload with meta info of the data.
            - shm_obj (:obj:`ShmObject`): The shm buffer.
        """
        assert isinstance(payload.data, type(
            shm_obj.buf
        )), "Data type ({}) and buf type ({}) are not match!".format(type(payload.data), type(shm_obj.buf))

        assert shm_obj.id_.get()[0] == payload.extra[0], "Shm object and payload do not match ({} - {}).".format(
            shm_obj.id_.get()[0], payload.extra[0]
        )

        def shm_putback(data: Union[Dict, List], buf: Union[Dict, List]):
            if isinstance(data, Dict) or isinstance(data, ttorch.Tensor):
                for key, val in data.items():
                    buf_val = buf.get(key)
                    if buf_val is None:
                        continue
                    if val is None and isinstance(buf_val, ShmBuffer):
                        data[key] = buf[key].get()
                    else:
                        shm_putback(val, buf_val)
            elif isinstance(data, List):
                for i, data_ in enumerate(data):
                    shm_putback(data_, buf[i])

        shm_putback(payload.data, buf=shm_obj.buf)
        shm_obj.id_.fill(np.array([0]))


class FileStorageLoader(StorageLoader):

    def __init__(self, dirname: str, ttl: int = 20, worker_num: int = 3) -> None:
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

    def save(self, obj: Union[Dict, List]) -> FileStorage:
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
