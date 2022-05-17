from copy import copy
import multiprocessing as mp
import threading
import queue
import platform
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union
from enum import Enum


@dataclass
class SendPayload:
    proc_id: int
    req_id: str = field(default_factory=lambda: uuid.uuid1().hex)
    method: str = None
    reserve_method: str = None
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


class Child:
    """
    Abstract class of child process/thread.
    """

    def __init__(self, proc_id: int, init: Callable, args: List[Any], recv_queue: Union[mp.Queue, queue.Queue]) -> None:
        self._proc_id = proc_id
        self._init = init
        self._args = args
        self._recv_queue = recv_queue

    def start(self):
        raise NotImplementedError

    def restart(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def send(self, payload: SendPayload):
        raise NotImplementedError

    def _target(
        self, proc_id: int, init: Callable, args: List, send_queue: Union[mp.Queue, queue.Queue],
        recv_queue: Union[mp.Queue, queue.Queue]
    ):
        send_payload = SendPayload(proc_id=proc_id)
        try:
            child_ins = init(*args)
            while True:
                send_payload: SendPayload = send_queue.get()
                if send_payload.reserve_method == ReserveMethod.SHUTDOWN:
                    break
                if send_payload.reserve_method == ReserveMethod.GETATTR:
                    data = getattr(child_ins, send_payload.args[0])
                else:
                    data = getattr(child_ins, send_payload.method)(*send_payload.args, **send_payload.kwargs)
                recv_queue.put(
                    RecvPayload(proc_id=proc_id, req_id=send_payload.req_id, method=send_payload.method, data=data)
                )
        except Exception as e:
            logging.error("Error in child process! id: {}, error: {}".format(self._proc_id, e))
            recv_payload = RecvPayload(proc_id=proc_id, req_id=send_payload.req_id, method=send_payload.method, err=e)
            recv_queue.put(recv_payload)

    def __del__(self):
        self.shutdown()


class ChildProcess(Child):

    def __init__(self, proc_id: int, init: Callable, args: List[Any], recv_queue: mp.Queue) -> None:
        super().__init__(proc_id, init, args, recv_queue)
        self._send_queue = mp.Queue()
        self._proc = None

    def start(self):
        context = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        ctx = mp.get_context(context)
        proc = ctx.Process(
            target=self._target,
            args=(self._proc_id, self._init, self._args, self._send_queue, self._recv_queue),
            daemon=True
        )
        proc.start()
        self._proc = proc

    def restart(self):
        self.shutdown()
        self.start()

    def shutdown(self):
        if self._proc:
            self._send_queue.put(SendPayload(proc_id=self._proc_id, method=ReserveMethod.SHUTDOWN))
            self._proc.terminate()
            self._proc.join()
            self._proc.close()
            self._proc = None
            self._send_queue.close()
            self._send_queue = mp.Queue()

    def send(self, payload: SendPayload):
        self._send_queue.put(payload)


class ChildThread(Child):

    def __init__(self, proc_id: int, init: Callable, args: List[Any], recv_queue: queue.Queue) -> None:
        super().__init__(proc_id, init, args, recv_queue)
        self._send_queue = queue.Queue()
        self._thread = None

    def start(self):
        thread = threading.Thread(
            target=self._target,
            args=(self._proc_id, self._init, self._args, self._send_queue, self._recv_queue),
            daemon=True
        )
        thread.start()
        self._thread = thread

    def restart(self):
        self.shutdown()
        self.start()

    def shutdown(self):
        if self._thread:
            self._send_queue.put(SendPayload(proc_id=self._proc_id, reserve_method=ReserveMethod.SHUTDOWN))
            self._thread.join()
            self._thread = None
            self._send_queue = queue.Queue()

    def send(self, payload: SendPayload):
        self._send_queue.put(payload)


class Supervisor:

    TYPE_MAPPING = {ChildType.PROCESS: ChildProcess, ChildType.THREAD: ChildThread}

    QUEUE_MAPPING = {ChildType.PROCESS: mp.Queue, ChildType.THREAD: queue.Queue}

    def __init__(self, type_: ChildType) -> None:
        self._children: List[Child] = []
        self._type = type_
        self._child_class = self.TYPE_MAPPING[self._type]
        self._recv_queue: queue.Queue = self.QUEUE_MAPPING[self._type]()

    def register(self, init: Callable, *args) -> None:
        proc_id = len(self._children)
        self._children.append(self._child_class(proc_id, init=init, args=args, recv_queue=self._recv_queue))

    def start_link(self) -> None:
        for child in self._children:
            child.start()

    def send(self, payload: SendPayload) -> None:
        """
        Overview:
            Send message to child process.
        Arguments:
            - payload (:obj:`SendPayload`): Send payload.
        """
        self._children[payload.proc_id].send(payload)

    def recv(self, ignore_err: bool = False) -> RecvPayload:
        """
        Overview:
            Wait for message from child process
        Arguments:
            - ignore_err (:obj:`bool`): If ignore_err is True, the child process will automatically restart, \
                and put the err in the property of recv_payload. Otherwise, an exception will be raised.
        Returns:
            - recv_payload (:obj:`RecvPayload`): Recv payload.
        """
        recv_payload: RecvPayload = self._recv_queue.get()
        if recv_payload.err:
            if ignore_err:
                self._children[recv_payload.proc_id].restart()
            else:
                raise recv_payload.err
        return recv_payload

    def recv_all(self, req_ids: List[str], ignore_err: bool = False) -> List[RecvPayload]:
        """
        Overview:
            Wait for messages with specific req ids until all ids are fulfilled.
        Arguments:
            - req_ids (:obj:`List[str]`): Request ids.
            - ignore_err (:obj:`bool`): If ignore_err is True, the child process will automatically restart, \
                and put the err in the property of recv_payload. Otherwise, an exception will be raised.
        Returns:
            - recv_payload (:obj:`RecvPayload`): Recv payload.
        """
        return_payloads = {}
        req_ids_copy = copy(req_ids)
        while req_ids_copy:
            recv_payload: RecvPayload = self._recv_queue.get()
            if recv_payload.req_id in req_ids_copy:
                req_ids_copy.remove(recv_payload.req_id)
                return_payloads[recv_payload.req_id] = recv_payload
                if recv_payload.err:
                    if ignore_err:
                        self._children[recv_payload.proc_id].restart()
                    else:
                        raise recv_payload.err
            else:
                # Put back the unrelated payload.
                self._recv_queue.put(recv_payload)
        # Keep the original order of requests.
        return [return_payloads[req_id] for req_id in req_ids]

    def shutdown(self) -> None:
        for child in self._children:
            child.shutdown()
        while not self._recv_queue.empty():
            self._recv_queue.get()

    def __getattr__(self, key: str) -> List[Any]:
        req_ids = []
        for i, child in enumerate(self._children):
            payload = SendPayload(proc_id=i, reserve_method=ReserveMethod.GETATTR, args=[key])
            req_ids.append(payload.req_id)
            child.send(payload)
        return [payload.data for payload in self.recv_all(req_ids)]

    def __del__(self) -> None:
        self.shutdown()
        self._children.clear()
