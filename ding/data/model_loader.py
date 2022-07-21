from abc import ABC, abstractmethod
import logging
from os import path
import os
from threading import Thread
from time import sleep, time
from typing import Callable, Optional
import uuid

import torch
from ding.data.storage.file import FileModelStorage
from ding.data.storage.storage import Storage
from ding.framework import Supervisor
from ding.framework.supervisor import ChildType, SendPayload


class ModelWorker():

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def save(self, storage: Storage) -> Storage:
        storage.save(self._model.state_dict())
        return storage


class ModelLoader(Supervisor, ABC):

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(type_=ChildType.PROCESS)
        self._model = model
        self._send_callback_loop = None
        self._send_callbacks = {}

    def start(self):
        if not self._running:
            self._model.share_memory()
            self.register(ModelWorker, self._model)
            self.start_link()
            self._send_callback_loop = Thread(target=self._loop_send_callback, daemon=True)
            self._send_callback_loop.start()

    def shutdown(self, timeout: Optional[float] = None) -> None:
        super().shutdown(timeout)
        self._send_callback_loop = None
        self._send_callbacks = {}

    def _loop_send_callback(self):
        while True:
            payload = self.recv(ignore_err=True)
            if payload.err:
                logging.warning("Got error when loading data: {}".format(payload.err))
                if payload.req_id in self._send_callbacks:
                    del self._send_callbacks[payload.req_id]
            else:
                if payload.req_id in self._send_callbacks:
                    callback = self._send_callbacks.pop(payload.req_id)
                    callback(payload.data)

    def load(self, storage: Storage) -> object:
        return storage.load()

    @abstractmethod
    def save(self, callback: Callable) -> Storage:
        """
        Overview:
            Save model asynchronously.
        """
        raise NotImplementedError


class FileModelLoader(ModelLoader):

    def __init__(self, model: torch.nn.Module, dirname: str, ttl: int = 60) -> None:
        super().__init__(model)
        self._dirname = dirname
        self._ttl = ttl
        self._files = []
        self._cleanup_thread = None

    def start(self):
        if not self._running:
            super().start()
            self._start_cleanup()

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
        while len(self._files) > 0:
            _, file_path = self._files.pop(0)
            if path.exists(file_path):
                os.remove(file_path)
        self._cleanup_thread = None

    def _loop_cleanup(self):
        while True:
            if len(self._files) == 0 or time() - self._files[0][0] < self._ttl:
                sleep(1)
                continue
            _, file_path = self._files.pop(0)
            if path.exists(file_path):
                os.remove(file_path)

    def save(self, callback: Callable) -> FileModelStorage:
        if not path.exists(self._dirname):
            os.mkdir(self._dirname)
        file_path = "model_{}.pth.tar".format(uuid.uuid1())
        file_path = path.join(self._dirname, file_path)
        model_storage = FileModelStorage(file_path)
        payload = SendPayload(proc_id=0, method="save", args=[model_storage])
        self.send(payload)
        self._send_callbacks[payload.req_id] = callback
        self._files.append([time(), file_path])
        return model_storage
