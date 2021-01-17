from abc import ABCMeta
from typing import Mapping, Any

from requests.exceptions import HTTPError

from ..base import get_values_from_response


class _IResponseInformation(metaclass=ABCMeta):

    @property
    def success(self) -> bool:
        raise NotImplementedError

    @property
    def code(self) -> int:
        raise NotImplementedError

    @property
    def message(self) -> str:
        raise NotImplementedError

    @property
    def data(self) -> Mapping[str, Any]:
        raise NotImplementedError


# exception class for processing response
class ResponseException(Exception, _IResponseInformation, metaclass=ABCMeta):

    def __init__(self, error: HTTPError):
        self.__error = error
        self.__status_code, self.__success, self.__code, self.__message, self.__data = \
            get_values_from_response(error.response)
        Exception.__init__(self, self.__message)

    @property
    def status_code(self) -> int:
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
