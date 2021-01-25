import math
from functools import wraps
from typing import Union, Callable

from .base import Validator, _reset_validate, _IValidator

number = _reset_validate(Validator(int) | float,
                         lambda v: TypeError('{int} or {float} expected but {actual} found'.format(
                             int=repr(int.__name__), float=repr(float.__name__), actual=repr(v.__class__.__name__)
                         )))

string = Validator(str)

STRING_PROCESSOR = Callable[[str], str]


def enum(*items, case_sensitive: bool = True, strip: bool = True) -> _IValidator:
    def _case_sensitive(func: STRING_PROCESSOR) -> STRING_PROCESSOR:
        if case_sensitive:
            return func
        else:
            @wraps(func)
            def _new_func(value: str) -> str:
                return func(value).lower()

            return _new_func

    def _strip(func: STRING_PROCESSOR) -> STRING_PROCESSOR:
        if strip:
            return func
        else:
            @wraps(func)
            def _new_func(value: str) -> str:
                return func(value).strip()

            return _new_func

    _item_process = _case_sensitive(_strip(lambda x: x))
    item_set = set([_item_process(item) for item in items])

    def _validate(value: str):
        real_value = _item_process(value)
        if real_value not in item_set:
            raise ValueError('unknown enum value {value}'.format(value=repr(value)))

    return Validator(_validate)


def numeric(int_ok: bool = True, float_ok: bool = True, inf_ok: bool = True) -> _IValidator:
    if not int_ok and not float_ok:
        raise ValueError('Either int or float should be allowed.')

    def _validate(value: Union[int, float, str]):
        if math.isnan(value):
            raise ValueError('nan is not numeric value')
        elif isinstance(value, str):
            try:
                if int_ok and float_ok:
                    try:
                        _value = int(value)
                    except TypeError:
                        _value = float(value)
                elif int_ok:
                    _value = int(value)
                else:  # float_ok
                    _value = float(value)
            except TypeError:
                raise ValueError('str numeric value should be a valid 10 based int or float')

            _validate(_value)
        elif isinstance(value, (int, float)):
            if isinstance(value, int) and not int_ok:
                raise TypeError('int is not allowed but {actual} found'.format(actual=repr(value)))
            if isinstance(value, float) and not float_ok:
                raise TypeError('float is not allowed but {actual} found'.format(actual=repr(value)))
            if math.isinf(value) and not inf_ok:
                raise ValueError('inf is not allowed but {actual} found'.format(actual=repr(value)))
        else:
            raise TypeError('numeric value should be either int, float or str, but {actual} found'.format(
                actual=repr(type(value).__name__)))

    return Validator(_validate)
