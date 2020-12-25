from typing import Optional, Any, Mapping

from .error_code import SlaveErrorCode
from ..base import ResponsibleException


class ConnectionRefuse(ResponsibleException):

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.SLAVE_CONNECTION_REFUSED,
            message='Connection refused!',
            data=data or {},
            status_code=403,
        )


class DisconnectionRefuse(ResponsibleException):

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.SLAVE_DISCONNECTION_REFUSED,
            message='Disconnection refused!',
            data=data or {},
            status_code=403,
        )


class TaskRefuse(ResponsibleException):

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.TASK_REFUSED,
            message='Task refused!',
            data=data or {},
            status_code=403,
        )


class TaskFail(Exception):

    def __init__(self, result: Optional[Mapping[str, Any]], message: Optional[str] = None):
        if message:
            Exception.__init__(self, 'Task process failed - {message}.'.format(message=message))
        else:
            Exception.__init__(self, 'Task process failed.')
        self.__result = result or {}

    @property
    def result(self) -> Mapping[str, Any]:
        return self.__result
