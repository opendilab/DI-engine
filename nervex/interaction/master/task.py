from enum import unique, IntEnum
from threading import Lock
from typing import Mapping, Any, Optional, Callable
from uuid import UUID, uuid4

import requests
from requests import RequestException

from .base import _BEFORE_HOOK_TYPE, _AFTER_HOOK_TYPE, _ERROR_HOOK_TYPE
from ..base import HttpEngine, get_values_from_response, default_func


@unique
class TaskResultType(IntEnum):
    COMPLETED = 1
    FAILED = 2


@unique
class TaskStatus(IntEnum):
    IDLE = 0x00

    STARTING = 0x11
    STARTED = 0x12
    START_FAILED = 0x13

    COMPLETED = 0x21
    FAILED = 0x22


_COMPLETE_TRIGGER_NAME = '__TASK_COMPLETE__'
_FAIL_TRIGGER_NAME = '__TASK_FAIL__'


class Task:

    def __init__(
        self,
        http_engine: HttpEngine,
        data: Mapping[str, Any],
        task_id: Optional[UUID] = None,
        before_task_start: Optional[_BEFORE_HOOK_TYPE] = None,
        after_task_start: Optional[_AFTER_HOOK_TYPE] = None,
        error_task_start: Optional[_ERROR_HOOK_TYPE] = None
    ):
        self.__http_engine = http_engine
        self.__lock = Lock()

        self.__task_id = task_id or uuid4()
        self.__task_data = data
        self.__task_result = None
        self.__task_status = TaskStatus.IDLE
        self.__task_lock = Lock()

        self.__before_task_start = before_task_start or (lambda d: d)
        self.__after_task_start = default_func(None)(after_task_start)
        self.__error_task_start = default_func(None)(error_task_start)
        self.__after_task_completed_callbacks = []
        self.__after_task_failed_callbacks = []

        self.__init_triggers()

    def __request(self, method: str, path: str, data: Optional[Mapping[str, Any]] = None) -> requests.Response:
        return self.__http_engine.request(method, path, data)

    def __task_start(self):
        try:
            self.__task_status = TaskStatus.STARTING
            response = self.__request(
                'POST', '/task/new', {
                    'task': {
                        'id': str(self.__task_id)
                    },
                    'data': self.__before_task_start(self.__task_data) or {}
                }
            )
        except RequestException as err:
            self.__task_status = TaskStatus.START_FAILED
            return self.__error_task_start(err)
        else:
            self.__task_status = TaskStatus.STARTED
            ret = self.__after_task_start(*get_values_from_response(response))
            self.__task_lock.acquire()
            return ret

    def __task_complete(self, result: Mapping[str, Any]):
        self.__task_status = TaskStatus.COMPLETED
        self.__task_result = result
        for _callback in self.__after_task_completed_callbacks:
            _callback(self.__task_data, result)
        self.__task_lock.release()

    def __task_fail(self, result: Mapping[str, Any]):
        self.__task_status = TaskStatus.FAILED
        self.__task_result = result
        for _callback in self.__after_task_failed_callbacks:
            _callback(self.__task_data, result)
        self.__task_lock.release()

    # trigger methods
    def __task_complete_trigger(self, result: Mapping[str, Any]):
        with self.__lock:
            if self.__task_status == TaskStatus.STARTED:
                self.__task_complete(result)
            else:
                raise ValueError(
                    "Only task with {expect} status can be completed, but {actual} found.".format(
                        expect=repr(TaskStatus.STARTED.name),
                        actual=repr(self.__task_status.name),
                    )
                )

    def __task_fail_trigger(self, result: Mapping[str, Any]):
        with self.__lock:
            if self.__task_status == TaskStatus.STARTED:
                self.__task_fail(result)
            else:
                raise ValueError(
                    "Only task with {expect} status can be failed, but {actual} found.".format(
                        expect=repr(TaskStatus.STARTED.name),
                        actual=repr(self.__task_status.name),
                    )
                )

    def __init_triggers(self):
        setattr(self, _COMPLETE_TRIGGER_NAME, self.__task_complete_trigger)
        setattr(self, _FAIL_TRIGGER_NAME, self.__task_fail_trigger)

    # public properties
    @property
    def status(self) -> TaskStatus:
        return self.__task_status

    @property
    def task(self) -> Mapping[str, Any]:
        return self.__task_data

    @property
    def result(self) -> Optional[Mapping[str, Any]]:
        return self.__task_result

    # public methods
    def start(self) -> 'Task':
        with self.__lock:
            if self.__task_status == TaskStatus.IDLE:
                self.__task_start()
                return self
            else:
                raise ValueError(
                    "Only task with {expect} status can be started, but {actual} found.".format(
                        expect=repr(TaskStatus.IDLE.name),
                        actual=repr(self.__task_status.name),
                    )
                )

    def join(self) -> 'Task':
        with self.__task_lock:
            return self

    def on_complete(self, callback: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]) -> 'Task':
        with self.__lock:
            self.__after_task_completed_callbacks.append(callback)
            return self

    def on_fail(self, callback: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]) -> 'Task':
        with self.__lock:
            self.__after_task_failed_callbacks.append(callback)
            return self


def _task_complete(task: Task, result: Mapping[str, Any]):
    getattr(task, _COMPLETE_TRIGGER_NAME)(result)


def _task_fail(task: Task, result: Mapping[str, Any]):
    getattr(task, _FAIL_TRIGGER_NAME)(result)
