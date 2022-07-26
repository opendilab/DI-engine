from abc import ABC, abstractmethod
import logging
from os import path
import os
from threading import Thread
from time import sleep, time
from typing import Callable, Optional
import uuid
import torch.multiprocessing as mp

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
        """
        Overview:
            Save and send models asynchronously and load them synchronously.
        Arguments:
            - model (:obj:`torch.nn.Module`): Torch module.
        """
        if next(model.parameters()).is_cuda:
            super().__init__(type_=ChildType.PROCESS, mp_ctx=mp.get_context("spawn"))
        else:
            super().__init__(type_=ChildType.PROCESS)
        self._model = model
        self._send_callback_loop = None
        self._send_callbacks = {}
        self._model_worker = ModelWorker(self._model)

    def start(self):
        if not self._running:
            self._model.share_memory()
            self.register(self._model_worker)
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
        """
        Overview:
            Load model synchronously.
        Arguments:
            - storage (:obj:`Stroage`): The model should be wrapped in a storage object, e.g. FileModelStorage.
        Returns:
            - object (:obj:): The loaded model.
        """
        return storage.load()

    @abstractmethod
    def save(self, callback: Callable) -> Storage:
        """
        Overview:
            Save model asynchronously.
        Arguments:
            - callback (:obj:`Callable`): The callback function after saving model.
        Returns:
            - storage (:obj:`Storage`): The storage object is created synchronously, so it can be returned.
        """
        raise NotImplementedError


class FileModelLoader(ModelLoader):

    def __init__(self, model: torch.nn.Module, dirname: str, ttl: int = 20) -> None:
        """
        Overview:
            Model loader using files as storage media.
        Arguments:
            - model (:obj:`torch.nn.Module`): Torch module.
            - dirname (:obj:`str`): The directory for saving files.
            - ttl (:obj:`int`): Files will be automatically cleaned after ttl. Note that \
                files that do not time out when the process is stopped are not cleaned up \
                (to avoid errors when other processes read the file), so you may need to \
                clean up the remaining files manually
        """
        super().__init__(model)
        self._dirname = dirname
        self._ttl = ttl
        self._files = []
        self._cleanup_thread = None

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

    def save(self, callback: Callable) -> FileModelStorage:
        if not self._running:
            logging.warning("Please start model loader before saving model.")
            return
        if not path.exists(self._dirname):
            os.mkdir(self._dirname)
        file_path = "model_{}.pth.tar".format(uuid.uuid1())
        file_path = path.join(self._dirname, file_path)
        model_storage = FileModelStorage(file_path)
        payload = SendPayload(proc_id=0, method="save", args=[model_storage])
        self.send(payload)

        def clean_callback(storage: Storage):
            self._files.append([time(), file_path])
            callback(storage)

        self._send_callbacks[payload.req_id] = clean_callback
        self._start_cleanup()
        return model_storage
