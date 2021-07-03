from typing import Optional, Any, Mapping

from ..base import ResponsibleException
from ..exception import SlaveErrorCode


class ConnectionRefuse(ResponsibleException):
    """
    Overview:
        Exception represents the refuse to connection to slave from master, can be used in method `_before_connection`.
    Example:
        - Without data

        >>> raise ConnectionRefuse

        - With refuse data

        >>> raise ConnectionRefuse({'data': 233})
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        """
        Overview:
            Constructor of ConnectionRefuse
        Arguments:
            - data (:obj:`Optional[Mapping[str, Any]]`): Key-value-formed refuse data
        """
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.SLAVE_CONNECTION_REFUSED,
            message='Connection refused!',
            data=data or {},
            status_code=403,
        )


class DisconnectionRefuse(ResponsibleException):
    """
    Overview:
        Exception represents the refuse to disconnection to slave from master,
        can be used in method `_before_disconnection`.
    Example:
        - Without data

        >>> raise DisconnectionRefuse

        - With refuse data

        >>> raise DisconnectionRefuse({'data': 233})
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        """
        Overview:
            Constructor of DisconnectionRefuse
        Arguments:
            - data (:obj:`Optional[Mapping[str, Any]]`): Key-value-formed refuse data
        """
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.SLAVE_DISCONNECTION_REFUSED,
            message='Disconnection refused!',
            data=data or {},
            status_code=403,
        )


class TaskRefuse(ResponsibleException):
    """
    Overview:
        Exception represents the refuse to tasks, can be used in method `_before_task`.
    Example:
        - Without data

        >>> raise TaskRefuse

        - With refuse data

        >>> raise TaskRefuse({'data': 233})
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        """
        Overview:
            Constructor of TaskRefuse
        Arguments:
            - data (:obj:`Optional[Mapping[str, Any]]`): Key-value-formed refuse data
        """
        ResponsibleException.__init__(
            self,
            SlaveErrorCode.TASK_REFUSED,
            message='Task refused!',
            data=data or {},
            status_code=403,
        )


class TaskFail(Exception):
    """
    Overview:
        Exception represents the failure of tasks, can be used in method `_process_task`.
    Example:
        - Without data

        >>> raise TaskFail

        - With failure data

        >>> raise TaskFail({'data': 233})

        - With both data and message

        >>> raise TaskFail({'data': 233}, 'this is message')
    """

    def __init__(self, result: Optional[Mapping[str, Any]] = None, message: Optional[str] = None):
        """
        Overview:
            Constructor of TaskFail
        Arguments:
            - result (:obj:`Optional[Mapping[str, Any]]`): Result of task failure
            - message (:obj:`Optional[str]`): Message of task failure
        """
        if message:
            Exception.__init__(self, 'Task process failed - {message}.'.format(message=message))
        else:
            Exception.__init__(self, 'Task process failed.')
        self.__result = result or {}

    @property
    def result(self) -> Mapping[str, Any]:
        """
        Overview:
            Get the result of task failure.
        Returns:
            Result of task failure.
        """
        return self.__result
