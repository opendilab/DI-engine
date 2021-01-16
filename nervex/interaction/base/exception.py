from abc import ABCMeta
from typing import Mapping, Any, Type, Optional, TypeVar

from requests.exceptions import HTTPError

from .app import get_values_from_response, CommonErrorCode, success_response, failure_response


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
class ResponseError(Exception, _IResponseInformation, metaclass=ABCMeta):

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


_TR = TypeVar('_TR', bound=ResponseError)


# noinspection PyTypeChecker
def get_response_error(class_name: str, err_type: Type[_TR]) -> Type[_TR]:
    return type(class_name, (err_type, ), {})


# exception class for processing request
class RequestException(Exception, _IResponseInformation, metaclass=ABCMeta):

    def __init__(
        self,
        success: bool,
        code: Optional[int] = None,
        message: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None
    ):
        self.__success = not not success
        self.__code = CommonErrorCode.SUCCESS if self.__success else (code or CommonErrorCode.COMMON_FAILURE)
        self.__message = str(message)
        self.__data = data or {}

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

    def response(self):
        if self.__success:
            return success_response(self.__data, self.__message)
        else:
            return failure_response(self.__code, self.__message, self.__data)


class RequestSuccess(RequestException, metaclass=ABCMeta):

    def __init__(self, data: Optional[Mapping[str, Any]] = None, message: Optional[str] = None):
        RequestException.__init__(self, True, CommonErrorCode.SUCCESS, message, data)


class RequestFail(RequestException, metaclass=ABCMeta):

    def __init__(
        self, code: Optional[int] = None, message: Optional[str] = None, data: Optional[Mapping[str, Any]] = None
    ):
        RequestException.__init__(self, False, code, message, data)
