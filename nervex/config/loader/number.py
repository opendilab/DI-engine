import math
from typing import Optional, Union

from .base import Loader, ILoaderClass

NUMBER_TYPES = (int, float)
NUMBER_TYPING = Union[int, float]


def numeric(int_ok: bool = True, float_ok: bool = True, inf_ok: bool = True) -> ILoaderClass:
    if not int_ok and not float_ok:
        raise ValueError('Either int or float should be allowed.')

    def _load(value: NUMBER_TYPING) -> NUMBER_TYPING:
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


def interval(left: Optional[NUMBER_TYPING] = None, right: Optional[NUMBER_TYPING] = None,
             left_ok: bool = True, right_ok: bool = True, eps: NUMBER_TYPING = 0.0) -> ILoaderClass:
    if left is None:
        left = -math.inf
    if right is None:
        right = +math.inf
    if left > right:
        raise ValueError("Left bound should no more than right bound, but {left} > {right}.".format(left=repr(left),
                                                                                                    right=repr(right)))
    eps = math.fabs(eps)

    def _value_compare_with_eps(a: NUMBER_TYPING, b: NUMBER_TYPING) -> int:
        if math.fabs(a - b) <= eps:
            return 0
        elif a < b:
            return -1
        else:
            return 1

    def _load(value: NUMBER_TYPING) -> NUMBER_TYPING:
        _left_check = _value_compare_with_eps(value, left)
        if _left_check < 0:
            raise ValueError(
                'value should be no less than {left} but {value} found'.format(left=repr(left), value=repr(value)))
        elif not left_ok and _left_check == 0:
            raise ValueError(
                'value should not be equal to left bound {left} but {value} found'.format(left=repr(left),
                                                                                          value=repr(value)))

        _right_check = _value_compare_with_eps(value, right)
        if _right_check > 0:
            raise ValueError(
                'value should be no more than {right} but {value} found'.format(right=repr(right), value=repr(value)))
        elif not right_ok and _right_check == 0:
            raise ValueError(
                'value should not be equal to right bound {right} but {value} found'.format(right=repr(right),
                                                                                            value=repr(value)))

        return value

    return Loader(_load)
