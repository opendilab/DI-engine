import math
from typing import Union, Optional

from .base import Validator, _IValidator

NUMBER_TYPING = Union[int, float]


def interval(left: Optional[NUMBER_TYPING] = None, right: Optional[NUMBER_TYPING] = None,
             left_ok: bool = True, right_ok: bool = True, eps: NUMBER_TYPING = 0.0) -> _IValidator:
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

    def _validate(value):
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

    return Validator(_validate)
