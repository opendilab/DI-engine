import json
import sys
import time
import traceback
from abc import abstractmethod
from functools import wraps
from threading import Thread, Event, Lock
from typing import Optional, Callable, Any, Mapping
from uuid import UUID

import requests
from flask import Flask, request

from .action import ConnectionRefuse, DisconnectionRefuse, TaskRefuse, TaskFail
from ..base import random_token, ControllableService, get_http_engine_class, split_http_address, success_response, \
    failure_response, DblEvent
from ..config import DEFAULT_SLAVE_PORT, DEFAULT_CHANNEL, GLOBAL_HOST, DEFAULT_HEARTBEAT_SPAN, MIN_HEARTBEAT_SPAN
from ..error import SlaveErrorCode


class Slave(ControllableService):

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        heartbeat_span: Optional[float] = None,
        channel: Optional[int] = None
    ):
        # server part
        self.__host = host or GLOBAL_HOST
        self.__port = port or DEFAULT_SLAVE_PORT
        self.__flask_app_value = None
        self.__run_app_thread = Thread(target=self.__run_app)

        # heartbeat part
        self.__heartbeat_span = max(heartbeat_span or DEFAULT_HEARTBEAT_SPAN, MIN_HEARTBEAT_SPAN)
        self.__heartbeat_thread = Thread(target=self.__heartbeat)

        # task part
        self.__has_task = DblEvent()
        self.__task_lock = Lock()
        self.__task_id = None
        self.__task_data = None
        self.__task_thread = Thread(target=self.__task)

        # self-connection part
        self.__self_http_engine = get_http_engine_class(headers={
            'Token': lambda: self.__self_token,
        })()('localhost', self.__port, False)
        self.__self_token = random_token()

        # master-connection part
        self.__channel = channel or DEFAULT_CHANNEL
        self.__connected = DblEvent()
        self.__master_token = None
        self.__master_address = None
        self.__master_http_engine = None

        # global part
        self.__shutdown_event = Event()
        self.__lock = Lock()

    # master connection
    def __register_master(self, token: str, address: str):
        self.__master_token = token
        self.__master_address = address
        self.__master_http_engine = get_http_engine_class(
            headers={
                'Channel': lambda: str(self.__channel),
                'Token': lambda: self.__master_token,
            }
        )()(*split_http_address(self.__master_address))

    def __unregister_master(self):
        self.__master_token = None
        self.__master_address = None
        self.__master_http_engine = None

    def __open_master_connection(self, token: str, address: str):
        self.__register_master(token, address)
        self.__connected.open()

    def __close_master_connection(self):
        self.__unregister_master()
        self.__connected.close()

    # server part
    def __generate_app(self):
        app = Flask(__name__)

        # master apis
        app.route('/connect', methods=['POST'])(self.__check_master_request(self.__connect, False))
        app.route('/disconnect', methods=['DELETE'])(self.__check_master_request(self.__disconnect, True))
        app.route('/task/new', methods=['POST'])(self.__check_master_request(self.__new_task, True))

        # self apis
        app.route('/ping', methods=['GET'])(self.__check_self_request(self.__self_ping))
        app.route('/shutdown', methods=['DELETE'])(self.__check_self_request(self.__self_shutdown))

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
                    code=SlaveErrorCode.SYSTEM_SHUTTING_DOWN, message='System has already been shutting down.'
                ), 401
            else:
                return func()

        return _func

    # server method checkers (master)
    def __check_master_request(self,
                               func: Callable[[str, Mapping[str, Any]], Any],
                               need_match: bool = True) -> Callable[[], Any]:
        return self.__check_shutdown(self.__check_channel(self.__check_master_token(func, need_match)))

    # noinspection DuplicatedCode
    def __check_channel(self, func: Callable[[], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            channel = request.headers.get('Channel', None)
            channel = int(channel) if channel else None

            if channel is None:
                return failure_response(code=SlaveErrorCode.CHANNEL_NOT_FOUND, message='Channel not found.'), 400
            elif channel != self.__channel:
                return failure_response(
                    code=SlaveErrorCode.CHANNEL_INVALID, message='Channel not match with this endpoint.'
                ), 403
            else:
                return func()

        return _func

    def __check_master_token(self,
                             func: Callable[[str, Mapping[str, Any]], Any],
                             need_match: bool = True) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            master_token = request.headers.get('Token', None)
            if master_token is None:
                return failure_response(
                    code=SlaveErrorCode.MASTER_TOKEN_NOT_FOUND, message='Master token not found.'
                ), 400
            elif need_match and (master_token != self.__master_token):
                return failure_response(
                    code=SlaveErrorCode.MASTER_TOKEN_INVALID, message='Master not match with this endpoint.'
                ), 403
            else:
                return func(master_token, json.loads(request.data.decode()))

        return _func

    # server method checkers (self)
    # noinspection DuplicatedCode
    def __check_self_request(self, func: Callable[[], Any]) -> Callable[[], Any]:
        return self.__check_shutdown(self.__check_slave_token(func))

    def __check_slave_token(self, func: Callable[[], Any]) -> Callable[[], Any]:

        @wraps(func)
        def _func():
            slave_token = request.headers.get('Token', None)

            if slave_token is None:
                return failure_response(code=SlaveErrorCode.SELF_TOKEN_NOT_FOUND, message='Slave token not found.'), 400
            elif slave_token != self.__self_token:
                return failure_response(
                    code=SlaveErrorCode.SELF_TOKEN_INVALID, message='Slave token not match with this endpoint.'
                ), 403
            else:
                return func()

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

    # server methods (master)
    # noinspection PyUnusedLocal
    def __connect(self, token: str, data: Mapping[str, Any]):
        if self.__connected.is_open():
            return failure_response(
                code=SlaveErrorCode.SLAVE_ALREADY_CONNECTED, message='This slave already connected.'
            ), 400
        else:
            _master_info, _connection_data = data['master'], data['data']

            try:
                self._before_connection(_connection_data)
            except ConnectionRefuse as err:
                return err.get_response()
            else:
                self.__open_master_connection(token, _master_info['address'])
                return success_response(message='Connect success.')

    # noinspection PyUnusedLocal
    def __new_task(self, token: str, data: Mapping[str, Any]):
        with self.__task_lock:
            if self.__has_task.is_open():
                return failure_response(code=SlaveErrorCode.TASK_ALREADY_EXIST, message='Already has a task.'), 400
            else:
                _task_info, _task_data = data['task'], data['data']
                _task_id = _task_info['id']

                try:
                    self._before_task(_task_data)
                except TaskRefuse as err:
                    return err.get_response()
                else:
                    self.__task_id = UUID(_task_id)
                    self.__task_data = _task_data
                    self.__has_task.open()
                    return success_response(message='Task received!')

    # noinspection PyUnusedLocal
    def __disconnect(self, token: str, data: Mapping[str, Any]):
        if self.__connected.is_close():
            return failure_response(
                code=SlaveErrorCode.SLAVE_NOT_CONNECTED, message='This slave not connected yet.'
            ), 400
        else:
            _disconnection_data = data['data']

            try:
                self._before_disconnection(_disconnection_data)
            except DisconnectionRefuse as err:
                return err.get_response()
            else:
                self.__close_master_connection()
                return success_response(message='Disconnect success.')

    # heartbeat part
    def __heartbeat(self):
        _last_time = time.time()
        while not self.__shutdown_event.is_set():
            if self.__connected.is_open():
                try:
                    self.__master_heartbeat()
                except requests.exceptions.RequestException as err:
                    self._lost_connection(self.__master_address, err)
                    self.__close_master_connection()
                    traceback.print_exception(*sys.exc_info(), file=sys.stderr)

            _last_time += self.__heartbeat_span
            time.sleep(_last_time - time.time())

    # task part
    def __task(self):
        while not self.__shutdown_event.is_set():
            self.__has_task.wait_for_open(timeout=1.0)
            if self.__has_task.is_open():
                # noinspection PyBroadException
                try:
                    result = self._process_task(self.__task_data)
                except TaskFail as fail:
                    self.__has_task.close()
                    self.__master_task_fail(fail.result)
                except Exception:
                    self.__has_task.close()
                    traceback.print_exception(*sys.exc_info(), file=sys.stderr)
                else:
                    self.__has_task.close()
                    self.__master_task_complete(result)

    # self request operations
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

    # master request operations
    def __master_request(
            self,
            method: Optional[str] = 'GET',
            path: Optional[str] = None,
            data: Optional[Mapping[str, Any]] = None
    ) -> requests.Response:
        return self.__master_http_engine.request(method, path, data)

    def __master_heartbeat(self):
        return self.__master_request('GET', '/slave/heartbeat')

    def __master_task_complete(self, result: Mapping[str, Any]):
        return self.__master_request(
            'PUT', '/slave/task/complete', data={
                'task': {
                    'id': str(self.__task_id)
                },
                'result': result or {},
            }
        )

    def __master_task_fail(self, result: Mapping[str, Any]):
        return self.__master_request(
            'PUT', '/slave/task/fail', data={
                'task': {
                    'id': str(self.__task_id)
                },
                'result': result or {},
            }
        )

    # public methods
    def ping(self) -> bool:
        with self.__lock:
            try:
                self.__ping_once()
            except (requests.exceptions.BaseHTTPError, requests.exceptions.RequestException):
                return False
            else:
                return True

    def start(self):
        with self.__lock:
            self.__task_thread.start()
            self.__heartbeat_thread.start()
            self.__run_app_thread.start()

            self.__ping_until_started()

    def shutdown(self):
        with self.__lock:
            self.__shutdown()

    def join(self):
        with self.__lock:
            self.__run_app_thread.join()
            self.__heartbeat_thread.join()
            self.__task_thread.join()

    # inherit method
    def _before_connection(self, data: Mapping[str, Any]):
        pass

    def _before_disconnection(self, data: Mapping[str, Any]):
        pass

    def _before_task(self, data: Mapping[str, Any]):
        pass

    def _lost_connection(self, master_address: str, err: requests.exceptions.RequestException):
        pass

    @abstractmethod
    def _process_task(self, task: Mapping[str, Any]):
        raise NotImplementedError
