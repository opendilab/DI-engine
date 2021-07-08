from abc import ABCMeta
from enum import unique, IntEnum
from typing import Type

import enum_tools
from requests import HTTPError

from .base import ResponseException
from ..base import get_values_from_response


@enum_tools.documentation.document_enum
@unique
class MasterErrorCode(IntEnum):
    """
    Overview:
        Error codes for master end
    """
    SUCCESS = 0  # doc: Master request success

    SYSTEM_SHUTTING_DOWN = 101  # doc: Master end is shutting down

    CHANNEL_NOT_GIVEN = 201  # doc: No channel id given in request
    CHANNEL_INVALID = 202  # doc: Channel id given not match with master end

    MASTER_TOKEN_NOT_GIVEN = 301  # doc: No master token found in connection request from slave
    MASTER_TOKEN_INVALID = 302  # doc: Master token auth failed in master end

    SELF_TOKEN_NOT_GIVEN = 401  # doc: No self token given in self request (such as ping, shutdown)
    SELF_TOKEN_INVALID = 402  # doc: Self token auth failed in master end itself (such as ping, shutdown)

    SLAVE_TOKEN_NOT_GIVEN = 501  # doc: No slave token given in service request from slave
    SLAVE_TOKEN_INVALID = 502  # doc: Slave token not found in master end

    TASK_DATA_INVALID = 601  # doc: Task data is invalid


# noinspection DuplicatedCode
class MasterResponseException(ResponseException, metaclass=ABCMeta):
    """
    Overview:
        Response exception for master client
    """

    def __init__(self, error: HTTPError):
        """
        Overview:
            Constructor
        Arguments:
            - error (:obj:`HTTPError`): Original http exception object
        """
        ResponseException.__init__(self, error)


class MasterSuccess(MasterResponseException):
    pass


class MasterSystemShuttingDown(MasterResponseException):
    pass


class MasterChannelNotGiven(MasterResponseException):
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


_PREFIX = ['master']


def get_master_exception_class_by_error_code(error_code: MasterErrorCode) -> Type[MasterResponseException]:
    """
    Overview:
        Transform from master error code to `MasterResponseException` class
    Arguments:
        - error_code (:obj:`MasterErrorCode`): Master error code
    Returns:
        - exception_class (:obj:`Type[MasterResponseException`): Master response exception class
    """
    class_name = ''.join([word.lower().capitalize() for word in (_PREFIX + error_code.name.split('_'))])
    return eval(class_name)


def get_master_exception_by_error(error: HTTPError) -> MasterResponseException:
    """
    Overview:
        Auto transform http error object to master response exception object.
    Arguments:
        - error (:obj:`HTTPError`): Http error object
    Returns:
        - exception (:obj:`MasterResponseException`): Master response exception object
    """
    _, _, code, _, _ = get_values_from_response(error.response)
    error_code = {v.value: v for k, v in MasterErrorCode.__members__.items()}[code]
    return get_master_exception_class_by_error_code(error_code)(error)
