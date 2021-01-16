from abc import ABCMeta
from enum import unique, IntEnum
from typing import Type

from requests import HTTPError

from .base import ResponseException
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


class MasterResponseException(ResponseException, metaclass=ABCMeta):

    def __init__(self, error: HTTPError):
        ResponseException.__init__(self, error)


class MasterSuccess(MasterResponseException):
    pass


class MasterSystemShuttingDown(MasterResponseException):
    pass


class MasterChannelNotFound(MasterResponseException):
    pass


class MasterChannelInvalid(MasterResponseException):
    pass


class MasterMasterTokenNotGiven(MasterResponseException):
    pass


class MasterMasterTokenInvalid(MasterResponseException):
    pass


class MasterSelfTokenNotGiven(MasterResponseException):
    pass


class MasterSelfTokenInvalid(MasterResponseException):
    pass


class MasterSlaveTokenNotGiven(MasterResponseException):
    pass


class MasterSlaveTokenInvalid(MasterResponseException):
    pass


class MasterTaskDataInvalid(MasterResponseException):
    pass


def get_exception_class_by_error_code(error_code: MasterErrorCode) -> Type[MasterResponseException]:
    class_name = ''.join([word.lower().capitalize() for word in error_code.name.split('_')])
    return eval(class_name)


def get_exception_by_error(error: HTTPError) -> MasterResponseException:
    _, _, code, _, _ = get_values_from_response(error.response)
    error_code = {v.value: v for k, v in MasterErrorCode.__members__.items()}[code]
    return get_exception_class_by_error_code(error_code)(error)
