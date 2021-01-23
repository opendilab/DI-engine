import json
import time
from functools import wraps, partial
from queue import Queue, Empty
from threading import Lock, Thread, Event
from typing import Optional, Any, Mapping, Type, Callable
from uuid import UUID

import requests
from flask import Flask, request
from requests.exceptions import RequestException
from urlobject import URLObject

from .connection import SlaveConnectionProxy, SlaveConnection, _ISlaveConnection, _get_connection_class, \
    _slave_task_complete, _slave_task_fail
from .task import TaskResultType
from ..base import random_token, ControllableService, failure_response, success_response, get_host_ip, \
    get_http_engine_class
from ..config import GLOBAL_HOST, DEFAULT_MASTER_PORT, DEFAULT_CHANNEL, MIN_HEARTBEAT_SPAN, DEFAULT_HEARTBEAT_SPAN, \
    DEFAULT_HEARTBEAT_TOLERANCE, MIN_HEARTBEAT_CHECK_SPAN, DEFAULT_HEARTBEAT_CHECK_SPAN
from ..exception import MasterErrorCode, get_master_exception_by_error


class Master(ControllableService):

    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None,
            heartbeat_span: Optional[float] = None,
            heartbeat_tolerance: Optional[float] = None,
            heartbeat_check_span: Optional[float] = None,
            channel: Optional[int] = None,
            my_address: Optional[str] = None
    ):
        # server part
        self.__host = host or GLOBAL_HOST
        self.__port = port or DEFAULT_MASTER_PORT
        self.__flask_app_value = None
        self.__run_app_thread = Thread(target=self.__run_app)

        # heartbeat part
        self.__heartbeat_span = max(heartbeat_span or DEFAULT_HEARTBEAT_SPAN, MIN_HEARTBEAT_SPAN)
        self.__heartbeat_tolerance = max(heartbeat_tolerance or DEFAULT_HEARTBEAT_TOLERANCE, MIN_HEARTBEAT_SPAN)
        self.__heartbeat_check_span = max(
            heartbeat_check_span or DEFAULT_HEARTBEAT_CHECK_SPAN, MIN_HEARTBEAT_CHECK_SPAN
        )
        self.__heartbeat_check_thread = Thread(target=self.__heartbeat_check)

        # self-connection part
        self.__self_http_engine = get_http_engine_class(
            headers={
                'Token': lambda: self.__self_token,
            },
            http_error_gene=get_master_exception_by_error,
        )()('localhost', self.__port, False)
        # )()(self.__host, self.__port, False)   # TODO: Confirm how to ping itself
        self.__self_token = random_token()

        # slave-connection part
        self.__channel = channel or DEFAULT_CHANNEL
        self.__my_address = my_address or str(
            URLObject().with_scheme('http').with_hostname(get_host_ip()).with_port(self.__port)
        )

        # slaves part
        self.__slaves = {}  # name --> (token, slave_connection)
        self.__token_slaves = {}  # token --> (name, slave_connection)
        self.__slave_last_heartbeat = {}  # name --> last_heartbeat
        self.__slave_lock = Lock()

        # task part
        self.__task_result_queue = Queue()
        self.__task_result_process_thread = Thread(target=self.__task_result_process)

        # global part
        self.__shutdown_event = Event()
        self.__lock = Lock()

    # slave connection
    def __connection_open(self, name: str, token: str, connection: SlaveConnectionProxy):
        with self.__slave_lock:
            self.__slaves[name] = (token, connection)
            self.__token_slaves[token] = (name, connection)
            self.__slave_last_heartbeat[name] = time.time()

    # noinspection PyUnusedLocal
    def __connection_close(self, name: str, connection: Optional[SlaveConnectionProxy] = None):
        with self.__slave_lock:
            token, _conn = self.__slaves[name]
            connection = connection or _conn
            del self.__slaves[name]
            del self.__token_slaves[token]
            del self.__slave_last_heartbeat[name]

    # server part
    def __generate_app(self):
        app = Flask(__name__)

        # self apis
        app.route('/ping', methods=['GET'])(self.__check_self_request(self.__self_ping))
        app.route('/shutdown', methods=['DELETE'])(self.__check_self_request(self.__self_shutdown))

        # slave apis
        app.route('/slave/heartbeat', methods=['GET'])(self.__check_slave_request(self.__heartbeat))
        app.route(
            '/slave/task/complete', methods=['PUT']
        )(self.__check_slave_request(self.__check_task_info(self.__task_complete)))
        app.route(
            '/slave/task/fail', methods=['PUT']
        )(self.__check_slave_request(self.__check_task_info(self.__task_fail)))

        return app

    def __flask_app(self) -> Flask:
        return self.__flask_app_value or self.__generate_app()

    def __run_app(self):
        self.__flask_app().run(
            host=self.__host,
            port=self.__port,
        )

    # both method checkers
    def __check_shutdown(self, func: Callable[[], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            if self.__shutdown_event.is_set():
                return failure_response(
                    code=MasterErrorCode.SYSTEM_SHUTTING_DOWN, message='System has already been shutting down.'
                ), 401
            else:
                return func()

        return _func

    # server method checkers (self)
    # noinspection DuplicatedCode
    def __check_self_request(self, func: Callable[[], Any]) -> Callable[[], Any]:
        return self.__check_shutdown(self.__check_master_token(func))

    def __check_master_token(self, func: Callable[[], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            master_token = request.headers.get('Token', None)

            if master_token is None:
                return failure_response(
                    code=MasterErrorCode.SELF_TOKEN_NOT_GIVEN, message='Master token not found.'
                ), 400
            elif master_token != self.__self_token:
                return failure_response(
                    code=MasterErrorCode.SELF_TOKEN_INVALID, message='Master token not match with this endpoint.'
                ), 403
            else:
                return func()

        return _func

    # server method checkers (slave)
    def __check_slave_request(self, func: Callable[[str, _ISlaveConnection], Any]) -> Callable[[], Any]:
        return self.__check_shutdown(self.__check_channel(self.__check_slave_token(func)))

    # noinspection DuplicatedCode
    def __check_channel(self, func: Callable[[], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            channel = request.headers.get('Channel', None)
            channel = int(channel) if channel else None

            if channel is None:
                return failure_response(code=MasterErrorCode.CHANNEL_NOT_GIVEN, message='Channel not found.'), 400
            elif channel != self.__channel:
                return failure_response(
                    code=MasterErrorCode.CHANNEL_INVALID, message='Channel not match with this endpoint.'
                ), 403
            else:
                return func()

        return _func

    def __check_slave_token(self, func: Callable[[str, _ISlaveConnection], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            slave_token = request.headers.get('Token', None)

            if slave_token is None:
                return failure_response(
                    code=MasterErrorCode.SLAVE_TOKEN_NOT_GIVEN, message='Slave token not found.'
                ), 400
            elif slave_token not in self.__token_slaves.keys():
                return failure_response(
                    code=MasterErrorCode.SLAVE_TOKEN_INVALID, message='No matching slave token found in this endpoint.'
                ), 403
            else:
                name, connection = self.__token_slaves[slave_token]
                return func(name, connection)

        return _func

    # noinspection PyMethodMayBeStatic
    def __get_request_data(self, func: Callable[[str, _ISlaveConnection, Mapping[str, Any]], Any]) \
            -> Callable[[str, _ISlaveConnection], Any]:

        @wraps(func)
        def _func(name: str, connection: _ISlaveConnection):
            _data = json.loads(request.data.decode())
            return func(name, connection, _data)

        return _func

    def __check_task_info(self, func: Callable[[str, _ISlaveConnection, UUID, Mapping[str, Any]], Any]) \
            -> Callable[[str, _ISlaveConnection], Any]:

        @wraps(func)
        @self.__get_request_data
        def _func(name: str, connection: _ISlaveConnection, data: Mapping[str, Any]):
            if 'task' not in data.keys():
                return failure_response(
                    code=MasterErrorCode.TASK_DATA_INVALID,
                    message='Task information not found.',
                )
            _task_info, _task_result = data['task'], data['result']

            if 'id' not in _task_info.keys():
                return failure_response(code=MasterErrorCode.TASK_DATA_INVALID, message='Task ID not found.')
            _task_id = UUID(_task_info['id'])

            return func(name, connection, _task_id, _task_result)

        return _func

    # server methods (self)
    # noinspection PyMethodMayBeStatic
    def __self_ping(self):
        return success_response(message='PONG!')

    def __self_shutdown(self):
        _shutdown_func = request.environ.get('werkzeug.server.shutdown')
        if _shutdown_func is None:
            raise RuntimeError('Not running with the Werkzeug Server')

        self.__shutdown_event.set()
        _shutdown_func()

        return success_response(message='Shutdown request received, this server will be down later.')

    # server methods (slave)
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def __heartbeat(self, name: str, connection: _ISlaveConnection):
        self.__slave_last_heartbeat[name] = time.time()
        return success_response(message='Received!')

    # noinspection PyUnusedLocal
    def __task_complete(self, name: str, connection: _ISlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
        self.__task_result_queue.put((TaskResultType.COMPLETED, (connection, task_id, task_result)))
        return success_response(message='Result received!')

    # noinspection PyUnusedLocal
    def __task_fail(self, name: str, connection: _ISlaveConnection, task_id: UUID, task_result: Mapping[str, Any]):
        self.__task_result_queue.put((TaskResultType.FAILED, (connection, task_id, task_result)))
        return success_response(message='Result received!')

    # self request
    def __self_request(self, method: Optional[str] = 'GET', path: Optional[str] = None) -> requests.Response:
        return self.__self_http_engine.request(method, path)

    def __ping_once(self):
        return self.__self_request('GET', '/ping')

    def __ping_until_started(self):
        while True:
            try:
                self.__ping_once()
            except (requests.exceptions.BaseHTTPError, requests.exceptions.RequestException):
                time.sleep(0.2)
            else:
                break

    def __shutdown(self):
        self.__self_request('DELETE', '/shutdown')

    # heartbeat part
    def __heartbeat_check(self):
        _last_time = time.time()
        while not self.__shutdown_event.is_set():
            _current_time = time.time()

            _common_names = set(self.__slaves.keys()) & set(self.__slave_last_heartbeat.keys())
            for name in _common_names:
                _, connection = self.__slaves[name]
                last_heartbeat = self.__slave_last_heartbeat[name]
                if _current_time - last_heartbeat > self.__heartbeat_tolerance:
                    self.__connection_close(name, connection)

            _last_time += self.__heartbeat_check_span
            time.sleep(_last_time - time.time())

    # task process part
    def __task_result_process(self):
        while not self.__task_result_queue.empty() or not self.__shutdown_event.is_set():
            try:
                _result = self.__task_result_queue.get(timeout=3.0)
            except Empty:
                continue
            else:
                _type, (_connection, _task_id, _task_result) = _result
                _trigger_func = _slave_task_complete if _type == TaskResultType.COMPLETED else _slave_task_fail
                _trigger_func(_connection, _task_id, _task_result)

    # connection part
    def __get_connection_class(self) -> Type[SlaveConnection]:
        return _get_connection_class(
            before_new_task=self._before_new_task,
            after_new_task=self._after_new_task,
            error_new_task=self._error_new_task,
            before_connect=self._before_connect,
            after_connect=self._after_connect,
            error_connect=self._error_connect,
            before_disconnect=self._before_disconnect,
            after_disconnect=self._after_disconnect,
            error_disconnect=self._error_disconnect,
        )

    def __get_new_connection(
            self, name: str, host: str, port: Optional[int] = None, https: bool = False
    ) -> SlaveConnectionProxy:
        if name in self.__slaves.keys():
            raise KeyError('Connection {name} already exist.'.format(name=repr(name)))
        else:
            slave_token = random_token()
            connection = self.__get_connection_class()(
                host=host,
                port=port,
                https=https,
                channel=self.__channel,
                my_address=self.__my_address,
                token=slave_token,
            )

            return SlaveConnectionProxy(
                connection=connection,
                after_connect=partial(self.__connection_open, name=name, token=slave_token),
                after_disconnect=partial(self.__connection_close, name=name),
            )

    # public properties
    @property
    def my_address(self) -> str:
        with self.__lock:
            return self.__my_address

    # public methods
    def ping(self) -> bool:
        with self.__lock:
            try:
                self.__ping_once()
            except (requests.exceptions.BaseHTTPError, requests.exceptions.RequestException):
                return False
            else:
                return True

    def new_connection(
            self, name: str, host: str, port: Optional[int] = None, https: bool = False
    ) -> SlaveConnectionProxy:
        with self.__lock:
            return self.__get_new_connection(name, host, port, https)

    def __contains__(self, name: str):
        with self.__lock:
            return name in self.__slaves.keys()

    def __getitem__(self, name: str):
        with self.__lock:
            if name in self.__slaves.keys():
                _token, _connection = self.__slaves[name]
                return _connection
            else:
                raise KeyError('Connection {name} not found.'.format(name=repr(name)))

    def __delitem__(self, name: str):
        with self.__lock:
            if name in self.__slaves.keys():
                _token, _connection = self.__slaves[name]
                _connection.disconnect()
            else:
                raise KeyError('Connection {name} not found.'.format(name=repr(name)))

    def start(self):
        with self.__lock:
            self.__task_result_process_thread.start()
            self.__heartbeat_check_thread.start()
            self.__run_app_thread.start()

            self.__ping_until_started()

    def shutdown(self):
        with self.__lock:
            self.__shutdown()

    def join(self):
        with self.__lock:
            self.__run_app_thread.join()
            self.__heartbeat_check_thread.join()
            self.__task_result_process_thread.join()

    # inherit methods
    def _before_connect(self) -> Mapping[str, Any]:
        pass

    def _after_connect(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass

    def _error_connect(self, error: RequestException):
        raise error

    def _before_disconnect(self) -> Mapping[str, Any]:
        pass

    def _after_disconnect(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass

    def _error_disconnect(self, error: RequestException):
        raise error

    # noinspection PyMethodMayBeStatic
    def _before_new_task(self, data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        return data or {}

    def _after_new_task(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        pass

    def _error_new_task(self, error: RequestException):
        raise error
