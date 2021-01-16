from abc import ABCMeta
from enum import unique, IntEnum
from typing import Type

from requests import HTTPError

from .base import RequestException
from ..base import get_values_from_response


@unique
class MasterErrorCode(IntEnum):
    SUCCESS = 0

    SYSTEM_SHUTTING_DOWN = 101

    CHANNEL_NOT_GIVEN = 201
    CHANNEL_INVALID = 202

    MASTER_TOKEN_NOT_GIVEN = 301
    MASTER_TOKEN_INVALID = 302

    SELF_TOKEN_NOT_GIVEN = 401
    SELF_TOKEN_INVALID = 402

    SLAVE_TOKEN_NOT_GIVEN = 501
    SLAVE_TOKEN_INVALID = 502

    TASK_DATA_INVALID = 601


class MasterRequestException(RequestException, metaclass=ABCMeta):

    def __init__(self, err: HTTPError):
        _, success, code, message, data = get_values_from_response(err.response)
        RequestException.__init__(self, success, code, message, data)


class MasterSuccess(MasterRequestException):
    pass


class MasterSystemShuttingDown(MasterRequestException):
    pass


class MasterChannelNotFound(MasterRequestException):
    pass


class MasterChannelInvalid(MasterRequestException):
    pass


class MasterMasterTokenNotGiven(MasterRequestException):
    pass


class MasterMasterTokenInvalid(MasterRequestException):
    pass


class MasterSelfTokenNotGiven(MasterRequestException):
    pass


class MasterSelfTokenInvalid(MasterRequestException):
    pass


class MasterSlaveTokenNotGiven(MasterRequestException):
    pass


class MasterSlaveTokenInvalid(MasterRequestException):
    pass


class MasterTaskDataInvalid(MasterRequestException):
    pass


def get_exception_class_by_error_code(error_code: MasterErrorCode) -> Type[MasterRequestException]:
    class_name = ''.join([word.lower().capitalize() for word in error_code.name.split('_')])
    return eval(class_name)


def get_exception_by_error(error: HTTPError) -> MasterRequestException:
    _, _, code, _, _ = get_values_from_response(error.response)
    error_code = {v.value: v for k, v in MasterErrorCode.__members__.items()}[code]
    return get_exception_class_by_error_code(error_code)(error)
