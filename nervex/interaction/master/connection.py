from abc import ABCMeta, abstractmethod
from functools import wraps
from threading import Lock
from typing import Optional, Any, Mapping, Type, Callable
from uuid import uuid4, UUID

import requests
from requests.exceptions import RequestException

from .base import _BEFORE_HOOK_TYPE, _AFTER_HOOK_TYPE, _ERROR_HOOK_TYPE
from .task import Task, _task_complete, _task_fail
from ..base import random_token, ControllableContext, get_http_engine_class, get_values_from_response
from ..config import DEFAULT_CHANNEL, DEFAULT_SLAVE_PORT, DEFAULT_REQUEST_RETRIES, DEFAULT_REQUEST_RETRY_WAITING
from ..exception import get_slave_exception_by_error

_COMPLETE_TRIGGER_NAME = '__TASK_COMPLETE__'
_FAIL_TRIGGER_NAME = '__TASK_FAIL__'


class _ISlaveConnection(ControllableContext, metaclass=ABCMeta):
    """
    Overview:
        Basic model of the connection classes, such as `SlaveConnection` and `SlaveConnectionProxy`, \
        which are used widely in interaction module.
    Example:
        - The following code shows a sample to correctly use slave connection
        >>> connection = master.new_connection('cnn1,', '127.0.0.1', 2333)
        >>> connection.connect()
        >>> try:
        >>>     pass # do anything you like
        >>> finally:
        >>>     connection.disconnect()

        - Another simple structure of the code above
        >>> with master.new_connection('cnn1,', '127.0.0.1', 2333) as connection:
        >>>     pass # do anything you like, connect and disconnect will be done automatically
    """

    @abstractmethod
    def connect(self):
        """
        Overview:
            Connect to slave end.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def disconnect(self):
        """
        Overview:
            Disconnect from slave end.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def new_task(self, data: Optional[Mapping[str, Any]] = None):
        """
        Overview:
            Send new task to slave end and receive task result from it.
        Arguments:
            - data (:obj:`Optional[Mapping[str, Any]]`): Data of the new task
        Returns:
            - result (:obj:`Mapping[str, Any]`): Result of the task processed by slave end
        """
        raise NotImplementedError  # pragma: no cover

    def start(self):
        """
        Overview:
            Alias for `connect`, for supporting context manager.
        """
        self.connect()

    def close(self):
        """
        Overview:
            Alias for `disconnect`, for support context manager.
        """
        self.disconnect()


class SlaveConnection(_ISlaveConnection, metaclass=ABCMeta):
    """
    Overview:
        Slave connection object, which need to directly interact with slave end.
    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        https: bool = False,
        channel: Optional[int] = None,
        my_address: Optional[str] = None,
        token: Optional[str] = None,
        request_retries: Optional[int] = None,
        request_retry_waiting: Optional[float] = None,
    ):
        """
        Overview:
            Constructor of `SlaveConnection`
        Arguments:
            - host (:obj:`str`): Host of the slave server
            - port (:obj:`Optional[int]`): Port of the slave server (None means `7236`)
            - https (:obj:`bool`): Use https or not
            - channel (:obj:`Optional[int]`): Channel id for the slave client.
            - my_address (:obj:`Optional[str]`): The address of current server (None will grep local ip automatically, \
                this address will be used when connect to slave, the slave's request will be send to this address, \
                **so please make sure the address can be achieved by slave**)
            - token (:obj:`Optional[str]`): Token of this connection, it is a token for authenticate to the \
                connection (`None` means this token would be randomly generated)
            - request_retries (:obj:`Optional[int]`): Max times for request retries (None means `5`)
            - request_retry_waiting (:obj:`Optional[float]`): Sleep time before requests' retrying (None means `1.0`, \
                unit: second)
        """
        # meta info part
        self.__channel = channel or DEFAULT_CHANNEL
        self.__my_address = my_address
        self.__token = token or random_token()

        # request part
        self.__http_engine = get_http_engine_class(
            headers={
                'Channel': lambda: str(self.__channel),
                'Token': lambda: self.__token,
            },
            http_error_gene=get_slave_exception_by_error,
        )()(host, port or DEFAULT_SLAVE_PORT, https)
        self.__request_retries = max(request_retries or DEFAULT_REQUEST_RETRIES, 0)
        self.__request_retry_waiting = max(request_retry_waiting or DEFAULT_REQUEST_RETRY_WAITING, 0.0)

        # threading part
        self.__lock = Lock()
        self.__is_connected = False

        # task part
        self.__tasks = {}

        self.__init_triggers()

    def __request(self, method: str, path: str, data: Optional[Mapping[str, Any]] = None) -> requests.Response:
        return self.__http_engine.request(
            method,
            path,
            data,
            retries=self.__request_retries,
            retry_waiting=self.__request_retry_waiting,
        )

    @property
    def is_connected(self) -> bool:
        """
        Overview:
            Check connection status
        Returns:
            - connected (:obj:`bool`): Whether this connection is still alive
        """
        with self.__lock:
            return self.__is_connected

    def _before_connect(self) -> Mapping[str, Any]:
        pass  # pragma: no cover

    def _after_connect(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass  # pragma: no cover

    def _error_connect(self, error: RequestException) -> Any:
        raise error  # pragma: no cover

    def __connect(self):
        try:
            response = self.__request(
                'POST', '/connect', {
                    'master': {
                        'address': self.__my_address,
                    },
                    'data': (self._before_connect() or {})
                }
            )
        except RequestException as err:
            return self._error_connect(err)
        else:
            self.__is_connected = True
            return self._after_connect(*get_values_from_response(response))

    def connect(self):
        with self.__lock:
            return self.__connect()

    def _before_disconnect(self) -> Mapping[str, Any]:
        pass  # pragma: no cover

    def _after_disconnect(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass  # pragma: no cover

    def _error_disconnect(self, error: RequestException) -> Any:
        raise error  # pragma: no cover

    def __disconnect(self):
        try:
            response = self.__request('DELETE', '/disconnect', {
                'data': self._before_disconnect() or {},
            })
        except RequestException as err:
            return self._error_disconnect(err)
        else:
            self.__is_connected = False
            return self._after_disconnect(*get_values_from_response(response))

    def disconnect(self):
        with self.__lock:
            return self.__disconnect()

    def _before_new_task(self, data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        return data  # pragma: no cover

    def _after_new_task(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass  # pragma: no cover

    def _error_new_task(self, error: RequestException) -> Any:
        raise error  # pragma: no cover

    def new_task(self, data: Optional[Mapping[str, Any]] = None) -> Task:
        with self.__lock:
            _uuid = uuid4()
            _task = Task(
                http_engine=self.__http_engine,
                data=data,
                task_id=_uuid,
                before_task_start=self._before_new_task,
                after_task_start=self._after_new_task,
                error_task_start=self._error_new_task,
            )

            self.__tasks[_uuid] = _task
            return _task

    def __task_complete(self, task_id: UUID, task_result: Mapping[str, Any]):
        _task = self.__tasks[task_id]
        _task_complete(_task, task_result)
        del self.__tasks[task_id]

    def __task_fail(self, task_id: UUID, task_result: Mapping[str, Any]):
        _task = self.__tasks[task_id]
        _task_fail(_task, task_result)
        del self.__tasks[task_id]

    def __task_complete_trigger(self, task_id: UUID, task_result: Mapping[str, Any]):
        with self.__lock:
            if task_id in self.__tasks.keys():
                return self.__task_complete(task_id, task_result)
            else:
                raise KeyError("Task {uuid} not found in this connection.".format(uuid=repr(str(task_id))))

    def __task_fail_trigger(self, task_id: UUID, task_result: Mapping[str, Any]):
        with self.__lock:
            if task_id in self.__tasks.keys():
                return self.__task_fail(task_id, task_result)
            else:
                raise KeyError("Task {uuid} not found in this connection.".format(uuid=repr(str(task_id))))

    def __init_triggers(self):
        setattr(self, _COMPLETE_TRIGGER_NAME, self.__task_complete_trigger)
        setattr(self, _FAIL_TRIGGER_NAME, self.__task_fail_trigger)


def _connection_task_complete(connection: SlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
    return getattr(connection, _COMPLETE_TRIGGER_NAME)(task_id, task_result)


def _connection_task_fail(connection: SlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
    return getattr(connection, _FAIL_TRIGGER_NAME)(task_id, task_result)


class SlaveConnectionProxy(_ISlaveConnection):
    """
    Overview:
        Proxy class for `SlaveConnection` class, which wraps the original methods.
    """

    def __init__(
        self,
        connection: SlaveConnection,
        after_connect: Optional[Callable] = None,
        after_disconnect: Optional[Callable] = None
    ):
        """
        Overview:
            Constructor of `SlaveConnectionProxy`
        Arguments:
            - connection (:obj:`SlaveConnection`): Slave connection object
            - after_connect (:obj:`Optional[Callable]`): Behaviour going to be executed after connection established
            - after_disconnect (:obj:`Optional[Callable]`): Behaviour going to be executed after connection killed
        """
        self.__connection = connection
        self.__lock = Lock()
        self.__after_connect = after_connect
        self.__after_disconnect = after_disconnect

        self.__init_triggers()

    @property
    def is_connected(self) -> bool:
        """
        Overview:
            Check connection status
        Returns:
            - connected (:obj:`bool`): Whether this connection is still alive
        """
        with self.__lock:
            return self.__connection.is_connected

    def connect(self):
        with self.__lock:
            result = self.__connection.connect()
            if self.__after_connect is not None:
                self.__after_connect(connection=self)
            return result

    def disconnect(self):
        with self.__lock:
            result = self.__connection.disconnect()
            if self.__after_disconnect is not None:
                self.__after_disconnect(connection=self)
            return result

    def new_task(self, data: Optional[Mapping[str, Any]] = None):
        with self.__lock:
            return self.__connection.new_task(data)

    def __task_complete_trigger(self, task_id: UUID, task_result: Mapping[str, Any]):
        with self.__lock:
            return _connection_task_complete(self.__connection, task_id, task_result)

    def __task_fail_trigger(self, task_id: UUID, task_result: Mapping[str, Any]):
        with self.__lock:
            return _connection_task_fail(self.__connection, task_id, task_result)

    def __init_triggers(self):
        setattr(self, _COMPLETE_TRIGGER_NAME, self.__task_complete_trigger)
        setattr(self, _FAIL_TRIGGER_NAME, self.__task_fail_trigger)


def _proxy_task_complete(proxy: SlaveConnectionProxy, task_id: UUID, task_result: Mapping[str, Any]):
    return getattr(proxy, _COMPLETE_TRIGGER_NAME)(task_id, task_result)


def _proxy_task_fail(proxy: SlaveConnectionProxy, task_id: UUID, task_result: Mapping[str, Any]):
    return getattr(proxy, _FAIL_TRIGGER_NAME)(task_id, task_result)


def _slave_task_complete(connection: _ISlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
    if isinstance(connection, SlaveConnection):
        return _connection_task_complete(connection, task_id, task_result)
    elif isinstance(connection, SlaveConnectionProxy):
        return _proxy_task_complete(connection, task_id, task_result)
    else:
        raise TypeError(
            "{expect1} or {expect2} expected, but {actual} found.".format(
                expect1=SlaveConnection.__name__,
                expect2=SlaveConnectionProxy.__name__,
                actual=type(connection).__name__,
            )
        )


def _slave_task_fail(connection: _ISlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
    if isinstance(connection, SlaveConnection):
        return _connection_task_fail(connection, task_id, task_result)
    elif isinstance(connection, SlaveConnectionProxy):
        return _proxy_task_fail(connection, task_id, task_result)
    else:
        raise TypeError(
            "{expect1} or {expect2} expected, but {actual} found.".format(
                expect1=SlaveConnection.__name__,
                expect2=SlaveConnectionProxy.__name__,
                actual=type(connection).__name__,
            )
        )


def _default_wrap(func: Callable) -> Callable:

    @wraps(func)
    def _new_func(*args, **kwargs):
        if func:
            return func(*args, **kwargs)
        else:
            return None

    return _new_func


def _get_connection_class(
        before_new_task: Optional[_BEFORE_HOOK_TYPE] = None,
        after_new_task: Optional[_AFTER_HOOK_TYPE] = None,
        error_new_task: Optional[_ERROR_HOOK_TYPE] = None,
        before_connect: Optional[_BEFORE_HOOK_TYPE] = None,
        after_connect: Optional[_AFTER_HOOK_TYPE] = None,
        error_connect: Optional[_ERROR_HOOK_TYPE] = None,
        before_disconnect: Optional[_BEFORE_HOOK_TYPE] = None,
        after_disconnect: Optional[_AFTER_HOOK_TYPE] = None,
        error_disconnect: Optional[_ERROR_HOOK_TYPE] = None,
) -> Type[SlaveConnection]:

    class _Connection(SlaveConnection):

        def _before_connect(self) -> Mapping[str, Any]:
            return _default_wrap(before_connect)() or {}

        def _after_connect(
                self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str,
                                                                                                                 Any]]
        ) -> Any:
            return _default_wrap(after_connect)(status_code, success, code, message, data)

        def _error_connect(self, error: RequestException) -> Any:
            return _default_wrap(error_connect)(error)

        def _before_disconnect(self) -> Mapping[str, Any]:
            return _default_wrap(before_disconnect)() or {}

        def _after_disconnect(
                self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str,
                                                                                                                 Any]]
        ) -> Any:
            return _default_wrap(after_disconnect)(status_code, success, code, message, data)

        def _error_disconnect(self, error: RequestException) -> Any:
            return _default_wrap(error_disconnect)(error)

        def _before_new_task(self, data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
            return _default_wrap(before_new_task)(data) or {}

        def _after_new_task(
                self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str,
                                                                                                                 Any]]
        ) -> Any:
            return _default_wrap(after_new_task)(status_code, success, code, message, data)

        def _error_new_task(self, error: RequestException) -> Any:
            return _default_wrap(error_new_task)(error)

    return _Connection
