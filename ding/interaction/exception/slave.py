from abc import ABCMeta
from enum import unique, IntEnum
from typing import Type

import enum_tools
from requests import HTTPError

from .base import ResponseException
from ..base import get_values_from_response


@enum_tools.documentation.document_enum
@unique
class SlaveErrorCode(IntEnum):
    """
    Overview:
        Error code for slave end
    """
    SUCCESS = 0  # doc: Slave request success

    SYSTEM_SHUTTING_DOWN = 101  # doc: Slave end is shutting down

    CHANNEL_NOT_FOUND = 201  # doc: No channel id given in request
    CHANNEL_INVALID = 202  # doc: Channel id given not match with slave end

    MASTER_TOKEN_NOT_FOUND = 301  # doc: No master token found in connection request from master
    MASTER_TOKEN_INVALID = 302  # doc: Master token auth failed in slave end

    SELF_TOKEN_NOT_FOUND = 401  # doc: No self token given in self request (such as ping, shutdown)
    SELF_TOKEN_INVALID = 402  # doc: Self token auth failed in slave end itself (such as ping, shutdown)

    SLAVE_ALREADY_CONNECTED = 501  # doc: Slave end has already connected to another master end
    SLAVE_NOT_CONNECTED = 502  # doc: Slave end not connected with master end yey
    SLAVE_CONNECTION_REFUSED = 503  # doc: Connection to slave end refused
    SLAVE_DISCONNECTION_REFUSED = 504  # doc: Disconnection to slave end refused

    TASK_ALREADY_EXIST = 601  # doc: Slave end is processing another task
    TASK_REFUSED = 602  # doc: Task for slave end refused


# noinspection DuplicatedCode
class SlaveResponseException(ResponseException, metaclass=ABCMeta):
    """
    Overview:
        Response exception for slave client
    """

    def __init__(self, error: HTTPError):
        """
        Overview:
            Constructor
        Arguments:
            - error (:obj:`HTTPError`): Original http exception object
        """
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
    """
    Overview:
        Transform from slave error code to `SlaveResponseException` class
    Arguments:
        - error_code (:obj:`SlaveErrorCode`): Slave error code
    Returns:
        - exception_class (:obj:`Type[SlaveResponseException`): Slave response exception class
    """
    class_name = ''.join([word.lower().capitalize() for word in (_PREFIX + error_code.name.split('_'))])
    return eval(class_name)


def get_slave_exception_by_error(error: HTTPError) -> SlaveResponseException:
    """
    Overview:
        Auto transform http error object to slave response exception object.
    Arguments:
        - error (:obj:`HTTPError`): Http error object
    Returns:
        - exception (:obj:`SlaveResponseException`): Slave response exception object
    """
    _, _, code, _, _ = get_values_from_response(error.response)
    error_code = {v.value: v for k, v in SlaveErrorCode.__members__.items()}[code]
    return get_slave_exception_class_by_error_code(error_code)(error)
