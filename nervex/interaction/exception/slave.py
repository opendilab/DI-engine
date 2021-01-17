from abc import ABCMeta
from enum import unique, IntEnum
from typing import Type

from requests import HTTPError

from .base import ResponseException
from ..base import get_values_from_response


@unique
class SlaveErrorCode(IntEnum):
    SUCCESS = 0

    SYSTEM_SHUTTING_DOWN = 101

    CHANNEL_NOT_FOUND = 201
    CHANNEL_INVALID = 202

    MASTER_TOKEN_NOT_FOUND = 301
    MASTER_TOKEN_INVALID = 302

    SELF_TOKEN_NOT_FOUND = 401
    SELF_TOKEN_INVALID = 402

    SLAVE_ALREADY_CONNECTED = 501
    SLAVE_NOT_CONNECTED = 502
    SLAVE_CONNECTION_REFUSED = 503
    SLAVE_DISCONNECTION_REFUSED = 504

    TASK_ALREADY_EXIST = 601
    TASK_REFUSED = 602


# noinspection DuplicatedCode
class SlaveResponseException(ResponseException, metaclass=ABCMeta):

    def __init__(self, error: HTTPError):
        ResponseException.__init__(self, error)


class SlaveSuccess(SlaveResponseException):
    pass


class SlaveSystemShuttingDown(SlaveResponseException):
    pass


class SlaveChannelNotFound(SlaveResponseException):
    pass


class SlaveChannelInvalid(SlaveResponseException):
    pass


class SlaveMasterTokenNotFound(SlaveResponseException):
    pass


class SlaveMasterTokenInvalid(SlaveResponseException):
    pass


class SlaveSelfTokenNotFound(SlaveResponseException):
    pass


class SlaveSelfTokenInvalid(SlaveResponseException):
    pass


class SlaveSlaveAlreadyConnected(SlaveResponseException):
    pass


class SlaveSlaveNotConnected(SlaveResponseException):
    pass


class SlaveSlaveConnectionRefused(SlaveResponseException):
    pass


class SlaveSlaveDisconnectionRefused(SlaveResponseException):
    pass


class SlaveTaskAlreadyExist(SlaveResponseException):
    pass


class SlaveTaskRefused(SlaveResponseException):
    pass


_PREFIX = ['slave']


def get_slave_exception_class_by_error_code(error_code: SlaveErrorCode) -> Type[SlaveResponseException]:
    class_name = ''.join([word.lower().capitalize() for word in (_PREFIX + error_code.name.split('_'))])
    return eval(class_name)


def get_slave_exception_by_error(error: HTTPError) -> SlaveResponseException:
    _, _, code, _, _ = get_values_from_response(error.response)
    error_code = {v.value: v for k, v in SlaveErrorCode.__members__.items()}[code]
    return get_slave_exception_class_by_error_code(error_code)(error)
