from abc import abstractmethod
from functools import wraps
from typing import Callable, Union, TypeVar

GENERAL_EXCEPTIONS = (TypeError, ValueError)
_EXCEPTIONS_TYPING = Union[TypeError, ValueError]

_ValueType = TypeVar('_ValueType')
_Checker = Callable[[_ValueType], bool]
_Validator = Callable[[_ValueType], None]
_ExceptionMaker = Callable[[_ValueType], _EXCEPTIONS_TYPING]


def _check_to_validate(check: _Checker, exception_maker: _ExceptionMaker) -> _Validator:

    @wraps(check)
    def _func(value: _ValueType):
        if not check(value):
            raise exception_maker(value)

    return _func


def _validate_to_check(validate: _Validator) -> _Checker:

    @wraps(validate)
    def _func(value: _ValueType):
        try:
            validate(value)
        except GENERAL_EXCEPTIONS:
            return False
        else:
            return True

    return _func


def _reset_validate(validate, exception_maker: _ExceptionMaker) -> '_IValidator':
    return _to_validator(_check_to_validate(_validate_to_check(_to_validator(validate).validate), exception_maker))


def _to_validator(validator) -> '_IValidator':
    if isinstance(validator, _IValidator):
        return validator
    elif isinstance(validator, tuple):
        _func, _message = validator
        return _to_validator(_check_to_validate(_func, _message))
    elif isinstance(validator, bool):
        return _to_validator((lambda v: validator, lambda v: ValueError('assertion false')))
    elif validator is None:
        return _to_validator(
            (
                lambda v: v is None,
                lambda v: TypeError('none expected but {value} found'.format(value=repr(v.__class__.__name__)))
            )
        )
    elif isinstance(validator, type):
        return _to_validator(
            (
                lambda v: isinstance(v, validator), lambda v: TypeError(
                    'type not match, {expect} expected '
                    'but {actual} found'.format(expect=repr(validator.__name__), actual=repr(v.__class__.__name__))
                )
            )
        )
    elif hasattr(validator, '__call__'):

        class _FuncValidator(_IValidator):

            def _validate(self, value) -> None:
                validator(value)

        return _FuncValidator()
    else:
        raise TypeError('Unknown type for validator generation.')


class _IValidator:

    @abstractmethod
    def _validate(self, value) -> None:
        raise NotImplementedError

    def __validate(self, value) -> None:
        self._validate(value)

    def __check(self, value) -> bool:
        try:
            self._validate(value)
        except (ValueError, TypeError):
            return False
        else:
            return True

    def validate(self, value) -> None:
        return self.__validate(value)

    def check(self, value) -> bool:
        return self.__check(value)

    def __call__(self, value) -> bool:
        return self.__check(value)

    def __and__(self, other) -> '_IValidator':

        def _validation(value):
            self.validate(value)
            _to_validator(other).validate(value)

        return _to_validator(_validation)

    def __rand__(self, other) -> '_IValidator':
        return _to_validator(other) & self

    def __or__(self, other) -> '_IValidator':

        def _validation(value):
            try:
                self.validate(value)
            except GENERAL_EXCEPTIONS:
                _to_validator(other).validate(value)

        return _to_validator(_validation)

    def __ror__(self, other) -> '_IValidator':
        return _to_validator(other) | self


Validator = _to_validator
