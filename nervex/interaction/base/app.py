import json
from enum import IntEnum, unique
from functools import wraps
from typing import Mapping, Any, Type, Optional, Tuple, Union, Iterable, Callable

import flask
import requests
from flask import jsonify
from requests.exceptions import HTTPError


@unique
class CommonErrorCode(IntEnum):
    SUCCESS = 0
    COMMON_FAILURE = 1


def flask_response(
        success: bool, data: Optional[Mapping[str, Any]] = None,
        message: Optional[str] = None, code: Optional[int] = None,
):
    return jsonify(
        {
            'success': success,
            'code': 0 if success else (code or CommonErrorCode.COMMON_FAILURE),
            'message': (message or 'Success.') if success else (message or 'Failed.'),
            'data': data,
        }
    )


def success_response(data: Optional[Mapping[str, Any]] = None, message: Optional[str] = None):
    return flask_response(
        success=True,
        code=CommonErrorCode.SUCCESS,
        message=message,
        data=data,
    )


def failure_response(
        code: Optional[int] = None, message: Optional[str] = None, data: Optional[Mapping[str, Any]] = None
):
    return flask_response(
        success=False,
        code=code or CommonErrorCode.COMMON_FAILURE,
        message=message,
        data=data,
    )


_RESPONSE_VALUE_TYPE = Tuple[int, bool, int, str, Mapping[str, Any]]


def get_values_from_response(response: Union[requests.Response, flask.Response]) -> _RESPONSE_VALUE_TYPE:
    status_code = response.status_code

    _content = response.content if hasattr(response, 'content') else response.data
    _json = json.loads(_content.decode())
    success, code, message, data = _json['success'], _json['code'], _json.get('message', ''), _json.get('data', {})

    return status_code, success, code, message, data


class ResponsibleException(Exception):

    def __init__(
            self,
            code: int = CommonErrorCode.COMMON_FAILURE,
            message: Optional[str] = None,
            data: Optional[Mapping[str, Any]] = None,
            status_code: int = 400
    ):
        Exception.__init__(self, message)
        self.__code = code
        self.__message = message
        self.__data = data or {}
        self.__status_code = status_code

    def get_response(self):
        return failure_response(self.__code, self.__message, self.__data), self.__status_code


def responsible(classes: Iterable[Type[ResponsibleException]] = None):
    if classes is None:
        classes = (ResponsibleException,)

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(func)
        def _func(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except tuple(classes) as err:
                return err.get_response()
            else:
                return ret

        return _func

    return _decorator


class ResponseError(Exception):
    def __init__(self, error: HTTPError):
        self._error = error
        self._status_code, self._success, self._code, self._message, self._data = \
            get_values_from_response(error.response)
        Exception.__init__(self, self._message)

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def success(self) -> bool:
        return self._success

    @property
    def code(self) -> int:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data


_TR = ResponseError


# noinspection PyTypeChecker
def get_request_error(class_name: str, err_type: Type[_TR]) -> Type[_TR]:
    return type(class_name, (err_type,), {})
