from abc import ABC, abstractmethod
import multiprocessing as mp
import threading
import queue
import platform
import traceback
import uuid
import time
from ditk import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum


@dataclass
class SendPayload:
    proc_id: int
    # Use uuid1 here to include the timestamp
    req_id: str = field(default_factory=lambda: uuid.uuid1().hex)
    method: str = None
    args: List = field(default_factory=list)
    kwargs: Dict = field(default_factory=dict)


@dataclass
class RecvPayload:
    proc_id: int
    req_id: str = None
    method: str = None
    data: Any = None
    err: Exception = None


class ReserveMethod(Enum):
    SHUTDOWN = "_shutdown"
    GETATTR = "_getattr"


class ChildType(Enum):
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class SharedObject:
    buf: Any
    callback: Callable


class Child(ABC):
    """
    Abstract class of child process/thread.
    """

    def __init__(
            self, proc_id: int, init: Callable, *args, shared_object: Optional[SharedObject] = None, **kwargs
    ) -> None:
        self._proc_id = proc_id
        self._init = init
        self._args = args
        self._kwargs = kwargs
        self._recv_queue = None
        self._send_queue = None
        self._shared_object = shared_object

    @abstractmethod
    def start(self, recv_queue: Union[mp.Queue, queue.Queue]):
        raise NotImplementedError

    def restart(self):
        self.shutdown()
        self.start(self._recv_queue)

    @abstractmethod
    def shutdown(self, timeout: Optional[float] = None):
        raise NotImplementedError

    @abstractmethod
    def send(self, payload: SendPayload):
        raise NotImplementedError

    def _target(
        self,
        proc_id: int,
        init: Callable,
        args: List,
        kwargs: Dict[str, Any],
        send_queue: Union[mp.Queue, queue.Queue],
        recv_queue: Union[mp.Queue, queue.Queue],
        shared_object: Optional[SharedObject] = None
    ):
        send_payload = SendPayload(proc_id=proc_id)
        child_ins = init(*args, **kwargs)
        while True:
            try:
                send_payload: SendPayload = send_queue.get()
                if send_payload.method == ReserveMethod.SHUTDOWN:
                    break
                if send_payload.method == ReserveMethod.GETATTR:
                    data = getattr(child_ins, send_payload.args[0])
                else:
                    data = getattr(child_ins, send_payload.method)(*send_payload.args, **send_payload.kwargs)
                recv_payload = RecvPayload(
                    proc_id=proc_id, req_id=send_payload.req_id, method=send_payload.method, data=data
                )
                if shared_object:
                    shared_object.callback(recv_payload, shared_object.buf)
                recv_queue.put(recv_payload)
            except Exception as e:
                logging.warning(traceback.format_exc())
                logging.warning("Error in child process! id: {}, error: {}".format(self._proc_id, e))
                recv_payload = RecvPayload(
                    proc_id=proc_id, req_id=send_payload.req_id, method=send_payload.method, err=e
                )
                recv_queue.put(recv_payload)

    def __del__(self):
        self.shutdown()


class ChildProcess(Child):

    def __init__(
            self, proc_id: int, init: Callable, *args, shared_object: Optional[SharedObject] = None, **kwargs
    ) -> None:
        super().__init__(proc_id, init, *args, shared_object=shared_object, **kwargs)
        self._proc = None

    def start(self, recv_queue: mp.Queue):
        self._recv_queue = recv_queue
        context = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        ctx = mp.get_context(context)
        self._send_queue = ctx.Queue()
        proc = ctx.Process(
            target=self._target,
            args=(
                self._proc_id, self._init, self._args, self._kwargs, self._send_queue, self._recv_queue,
                self._shared_object
            ),
            name="supervisor_child_{}_{}".format(self._proc_id, time.time()),
            daemon=True
        )
        proc.start()
        self._proc = proc

    def shutdown(self, timeout: Optional[float] = None):
        if self._proc:
            self._send_queue.put(SendPayload(proc_id=self._proc_id, method=ReserveMethod.SHUTDOWN))
            self._proc.terminate()
            self._proc.join(timeout=timeout)
            if hasattr(self._proc, "close"):  # Compatible with 3.6
                self._proc.close()
            self._proc = None
            self._send_queue.close()
            self._send_queue.join_thread()
            self._send_queue = None

    def send(self, payload: SendPayload):
        self._send_queue.put(payload)


class ChildThread(Child):

    def __init__(
            self, proc_id: int, init: Callable, *args, shared_object: Optional[SharedObject] = None, **kwargs
    ) -> None:
        super().__init__(proc_id, init, *args, shared_object=shared_object, **kwargs)
        self._thread = None

    def start(self, recv_queue: queue.Queue):
        self._recv_queue = recv_queue
        self._send_queue = queue.Queue()
        thread = threading.Thread(
            target=self._target,
            args=(self._proc_id, self._init, self._args, self._kwargs, self._send_queue, self._recv_queue),
            name="supervisor_child_{}_{}".format(self._proc_id, time.time()),
            daemon=True
        )
        thread.start()
        self._thread = thread

    def shutdown(self, timeout: Optional[float] = None):
        if self._thread:
            self._send_queue.put(SendPayload(proc_id=self._proc_id, method=ReserveMethod.SHUTDOWN))
            self._thread.join(timeout=timeout)
            self._thread = None
            self._send_queue = None

    def send(self, payload: SendPayload):
        self._send_queue.put(payload)


class Supervisor:

    TYPE_MAPPING = {ChildType.PROCESS: ChildProcess, ChildType.THREAD: ChildThread}

    QUEUE_MAPPING = {
        ChildType.PROCESS: mp.get_context('spawn' if platform.system().lower() == 'windows' else 'fork').Queue,
        ChildType.THREAD: queue.Queue
    }

    def __init__(self, type_: ChildType) -> None:
        self._children: List[Child] = []
        self._type = type_
        self._child_class = self.TYPE_MAPPING[self._type]
        self._running = False
        self.__queue = None

    def register(self, init: Callable, *args, shared_object: Optional[SharedObject] = None, **kwargs) -> None:
        proc_id = len(self._children)
        self._children.append(self._child_class(proc_id, init, *args, shared_object=shared_object, **kwargs))

    @property
    def _recv_queue(self) -> Union[queue.Queue, mp.Queue]:
        if not self.__queue:
            self.__queue = self.QUEUE_MAPPING[self._type]()
        return self.__queue

    @_recv_queue.setter
    def _recv_queue(self, queue: Union[queue.Queue, mp.Queue]):
        self.__queue = queue

    def start_link(self) -> None:
        if not self._running:
            for child in self._children:
                child.start(recv_queue=self._recv_queue)
            self._running = True

    def send(self, payload: SendPayload) -> None:
        """
        Overview:
            Send message to child process.
        Arguments:
            - payload (:obj:`SendPayload`): Send payload.
        """
        self._children[payload.proc_id].send(payload)

    def recv(self, ignore_err: bool = False, timeout: float = None) -> RecvPayload:
        """
        Overview:
            Wait for message from child process
        Arguments:
            - ignore_err (:obj:`bool`): If ignore_err is True, put the err in the property of recv_payload. \
                Otherwise, an exception will be raised.
            - timeout (:obj:`float`): Timeout for queue.get, will raise an Empty exception if timeout.
        Returns:
            - recv_payload (:obj:`RecvPayload`): Recv payload.
        """
        recv_payload: RecvPayload = self._recv_queue.get(timeout=timeout)
        if recv_payload.err and not ignore_err:
            raise recv_payload.err
        return recv_payload

    def recv_all(
            self,
            send_payloads: List[SendPayload],
            ignore_err: bool = False,
            callback: Callable = None,
            timeout: Optional[float] = None
    ) -> List[RecvPayload]:
        """
        Overview:
            Wait for messages with specific req ids until all ids are fulfilled.
        Arguments:
            - send_payloads (:obj:`List[SendPayload]`): Request payloads.
            - ignore_err (:obj:`bool`): If ignore_err is True, \
                put the err in the property of recv_payload. Otherwise, an exception will be raised. \
                This option will also ignore timeout error.
            - callback (:obj:`Callable`): Callback for each recv payload.
            - timeout (:obj:`Optional[float]`): Timeout when wait for responses.
        Returns:
            - recv_payload (:obj:`List[RecvPayload]`): Recv payload, may contain timeout error.
        """
        assert send_payloads, "Req payload is empty!"
        recv_payloads = {}
        remain_payloads = {payload.req_id: payload for payload in send_payloads}
        unrelated_payloads = []
        try:
            while remain_payloads:
                try:
                    recv_payload: RecvPayload = self._recv_queue.get(block=True, timeout=timeout)
                    if recv_payload.req_id in remain_payloads:
                        del remain_payloads[recv_payload.req_id]
                        recv_payloads[recv_payload.req_id] = recv_payload
                        if recv_payload.err and not ignore_err:
                            raise recv_payload.err
                        if callback:
                            callback(recv_payload, remain_payloads)
                    else:
                        unrelated_payloads.append(recv_payload)
                except queue.Empty:
                    if ignore_err:
                        req_ids = list(remain_payloads.keys())
                        logging.warning("Timeout ({}s) when receving payloads! Req ids: {}".format(timeout, req_ids))
                        for req_id in req_ids:
                            send_payload = remain_payloads.pop(req_id)
                            # If timeout error happens in timeout recover, there may not find any send_payload
                            # in the original indexed payloads.
                            recv_payload = RecvPayload(
                                proc_id=send_payload.proc_id,
                                req_id=send_payload.req_id,
                                method=send_payload.method,
                                err=TimeoutError("Timeout on req_id ({})".format(req_id))
                            )
                            recv_payloads[req_id] = recv_payload
                            if callback:
                                callback(recv_payload, remain_payloads)
                    else:
                        raise TimeoutError("Timeout ({}s) when receving payloads!".format(timeout))
        finally:
            # Put back the unrelated payload.
            for payload in unrelated_payloads:
                self._recv_queue.put(payload)

        # Keep the original order of requests.
        return [recv_payloads[p.req_id] for p in send_payloads]

    def shutdown(self, timeout: Optional[float] = None) -> None:
        if self._running:
            for child in self._children:
                child.shutdown(timeout=timeout)
            self._cleanup_queue()
        self._running = False

    def _cleanup_queue(self):
        while True:
            while not self._recv_queue.empty():
                self._recv_queue.get()
            time.sleep(0.1)  # mp.Queue is not reliable.
            if self._recv_queue.empty():
                break
        if hasattr(self._recv_queue, "close"):
            self._recv_queue.close()
            self._recv_queue.join_thread()
            self._recv_queue = None

    def __getattr__(self, key: str) -> List[Any]:
        assert self._running, "Supervisor is not running, please call start_link first!"
        send_payloads = []
        for i, child in enumerate(self._children):
            payload = SendPayload(proc_id=i, method=ReserveMethod.GETATTR, args=[key])
            send_payloads.append(payload)
            child.send(payload)
        return [payload.data for payload in self.recv_all(send_payloads)]

    def get_child_attr(self, proc_id: str, key: str) -> Any:
        """
        Overview:
            Get attr of one child process instance.
        Arguments:
            - proc_id (:obj:`str`): Proc id.
            - key (:obj:`str`): Attribute key.
        Returns:
            - attr (:obj:`Any`): Attribute of child.
        """
        assert self._running, "Supervisor is not running, please call start_link first!"
        payload = SendPayload(proc_id=proc_id, method=ReserveMethod.GETATTR, args=[key])
        self._children[proc_id].send(payload)
        payloads = self.recv_all([payload])
        return payloads[0].data

    def __del__(self) -> None:
        self.shutdown(timeout=5)
        self._children.clear()
