from abc import ABCMeta
from typing import Mapping, Any

from requests.exceptions import HTTPError

from ..base import get_values_from_response


class _IResponseInformation(metaclass=ABCMeta):
    """
    Overview:
        Response information basic structure interface
    """

    @property
    def success(self) -> bool:
        """
        Overview:
            Get response success or not
        Returns:
            - success (:obj:`bool`): Response success or not
        """
        raise NotImplementedError

    @property
    def code(self) -> int:
        """
        Overview:
            Get response error code (`0` means success)
        Returns:
            - code (:obj:`int`): Response error code
        """
        raise NotImplementedError

    @property
    def message(self) -> str:
        """
        Overview:
            Get response message
        Returns:
            - message (:obj:`str`): Response message
        """
        raise NotImplementedError

    @property
    def data(self) -> Mapping[str, Any]:
        """
        Overview:
            Get response data
        Returns:
            - data (:obj:`Mapping[str, Any]`): Response data
        """
        raise NotImplementedError


# exception class for processing response
class ResponseException(Exception, _IResponseInformation, metaclass=ABCMeta):
    """
    Overview:
        Response exception, which can be directly raised in methods to create fail http response.
    """

    def __init__(self, error: HTTPError):
        """
        Overview:
            Constructor of `ResponseException`
        Arguments:
            - error (:obj:`HTTPError`): Original http exception object
        """
        self.__error = error
        self.__status_code, self.__success, self.__code, self.__message, self.__data = \
            get_values_from_response(error.response)
        Exception.__init__(self, self.__message)

    @property
    def status_code(self) -> int:
        """
        Overview:
            Get http status code of response
        Returns:
            - status_code (:obj:`int`): Http status code
        """
        return self.__status_code

    @property
    def success(self) -> bool:
        return self.__success

    @property
    def code(self) -> int:
        return self.__code

    @property
    def message(self) -> str:
        return self.__message

    @property
    def data(self) -> Mapping[str, Any]:
        return self.__data
