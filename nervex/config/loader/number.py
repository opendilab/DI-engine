import math
import operator
from typing import Optional, Union, Callable, Any

from .base import Loader, ILoaderClass
from .utils import keep, check_only

NUMBER_TYPES = (int, float)
NUMBER_TYPING = Union[int, float]


def numeric(int_ok: bool = True, float_ok: bool = True, inf_ok: bool = True) -> ILoaderClass:
    if not int_ok and not float_ok:
        raise ValueError('Either int or float should be allowed.')

    def _load(value) -> NUMBER_TYPING:
        if isinstance(value, NUMBER_TYPES):
            if math.isnan(value):
                raise ValueError('nan is not numeric value')
            if isinstance(value, int) and not int_ok:
                raise TypeError('int is not allowed but {actual} found'.format(actual=repr(value)))
            if isinstance(value, float) and not float_ok:
                raise TypeError('float is not allowed but {actual} found'.format(actual=repr(value)))
            if math.isinf(value) and not inf_ok:
                raise ValueError('inf is not allowed but {actual} found'.format(actual=repr(value)))

            return value
        else:
            raise TypeError(
                'numeric value should be either int, float or str, but {actual} found'.format(
                    actual=repr(type(value).__name__)
                )
            )

    return Loader(_load)


def interval(
        left: Optional[NUMBER_TYPING] = None,
        right: Optional[NUMBER_TYPING] = None,
        left_ok: bool = True,
        right_ok: bool = True,
        eps=0.0
) -> ILoaderClass:
    if left is None:
        left = -math.inf
    if right is None:
        right = +math.inf
    if left > right:
        raise ValueError(
            "Left bound should no more than right bound, but {left} > {right}.".format(
                left=repr(left), right=repr(right)
            )
        )
    eps = math.fabs(eps)

    def _value_compare_with_eps(a, b) -> int:
        if math.fabs(a - b) <= eps:
            return 0
        elif a < b:
            return -1
        else:
            return 1

    def _load(value) -> NUMBER_TYPING:
        _left_check = _value_compare_with_eps(value, left)
        if _left_check < 0:
            raise ValueError(
                'value should be no less than {left} but {value} found'.format(left=repr(left), value=repr(value))
            )
        elif not left_ok and _left_check == 0:
            raise ValueError(
                'value should not be equal to left bound {left} but {value} found'.format(
                    left=repr(left), value=repr(value)
                )
            )

        _right_check = _value_compare_with_eps(value, right)
        if _right_check > 0:
            raise ValueError(
                'value should be no more than {right} but {value} found'.format(right=repr(right), value=repr(value))
            )
        elif not right_ok and _right_check == 0:
            raise ValueError(
                'value should not be equal to right bound {right} but {value} found'.format(
                    right=repr(right), value=repr(value)
                )
            )

        return value

    return Loader(_load)


def is_negative() -> ILoaderClass:
    return Loader((lambda x: x < 0, lambda x: ValueError('negative required but {value} found'.format(value=repr(x)))))


def is_positive() -> ILoaderClass:
    return Loader((lambda x: x > 0, lambda x: ValueError('positive required but {value} found'.format(value=repr(x)))))


def non_negative() -> ILoaderClass:
    return Loader(
        (lambda x: x >= 0, lambda x: ValueError('non-negative required but {value} found'.format(value=repr(x))))
    )


def non_positive() -> ILoaderClass:
    return Loader(
        (lambda x: x <= 0, lambda x: ValueError('non-positive required but {value} found'.format(value=repr(x))))
    )


def negative() -> ILoaderClass:
    return Loader(lambda x: -x)


def positive() -> ILoaderClass:
    return Loader(lambda x: +x)


def _math_binary(func: Callable[[Any, Any], Any], attachment) -> ILoaderClass:
    return Loader(lambda x: func(x, Loader(attachment)(x)))


def plus(addend) -> ILoaderClass:
    return _math_binary(lambda x, y: x + y, addend)


def minus(subtrahend) -> ILoaderClass:
    return _math_binary(lambda x, y: x - y, subtrahend)


def minus_with(minuend) -> ILoaderClass:
    return _math_binary(lambda x, y: y - x, minuend)


def multi(multiplier) -> ILoaderClass:
    return _math_binary(lambda x, y: x * y, multiplier)


def divide(divisor) -> ILoaderClass:
    return _math_binary(lambda x, y: x / y, divisor)


def divide_with(dividend) -> ILoaderClass:
    return _math_binary(lambda x, y: y / x, dividend)


def power(index) -> ILoaderClass:
    return _math_binary(lambda x, y: x ** y, index)


def power_with(base) -> ILoaderClass:
    return _math_binary(lambda x, y: y ** x, base)


def msum(*items) -> ILoaderClass:

    def _load(value):
        return sum([item(value) for item in items])

    return Loader(_load)


def mmulti(*items) -> ILoaderClass:

    def _load(value):
        _result = 1
        for item in items:
            _result *= item(value)
        return _result

    return Loader(_load)


_COMPARE_OPERATORS = {
    '!=': operator.__ne__,
    '==': operator.__eq__,
    '<': operator.__lt__,
    '<=': operator.__le__,
    '>': operator.__gt__,
    '>=': operator.__ge__,
}


def _msinglecmp(first, op, second) -> ILoaderClass:
    first = Loader(first)
    second = Loader(second)

    if op in _COMPARE_OPERATORS.keys():
        return Loader(
            (
                lambda x: _COMPARE_OPERATORS[op](first(x), second(x)), lambda x: ValueError(
                    'comparison failed for {first} {op} {second}'.format(
                        first=repr(first(x)),
                        second=repr(second(x)),
                        op=op,
                    )
                )
            )
        )
    else:
        raise KeyError('Invalid compare operator - {op}.'.format(op=repr(op)))


def mcmp(first, *items) -> ILoaderClass:
    if len(items) % 2 == 1:
        raise ValueError('Count of items should be odd number but {number} found.'.format(number=len(items) + 1))

    ops, items = items[0::2], items[1::2]

    _result = keep()
    for first, op, second in zip((first, ) + items[:-1], ops, items):
        _result &= _msinglecmp(first, op, second)

    return check_only(_result)
