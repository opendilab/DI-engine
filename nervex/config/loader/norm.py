import operator
from abc import abstractmethod
from functools import wraps
from typing import Callable, Any

from .base import ILoaderClass


def _callable_to_norm(func: Callable[[Any], Any]) -> 'INormClass':

    class _Norm(INormClass):

        def _call(self, value):
            return func(value)

    return _Norm()


def norm(value) -> 'INormClass':
    if isinstance(value, INormClass):
        return value
    elif isinstance(value, ILoaderClass):
        return _callable_to_norm(value)
    else:
        return _callable_to_norm(lambda v: value)


def normfunc(func):

    @wraps(func)
    def _new_func(*args_norm, **kwargs_norm):
        args_norm = [norm(item) for item in args_norm]
        kwargs_norm = {key: norm(value) for key, value in kwargs_norm.items()}

        def _callable(v):
            args = [item(v) for item in args_norm]
            kwargs = {key: value(v) for key, value in kwargs_norm.items()}
            return func(*args, **kwargs)

        return _callable_to_norm(_callable)

    return _new_func


UNARY_FUNC = Callable[[Any], Any]
BINARY_FUNC = Callable[[Any, Any], Any]


def _unary(a: 'INormClass', func: UNARY_FUNC) -> 'INormClass':
    return _callable_to_norm(lambda v: func(a(v)))


def _binary(a: 'INormClass', b: 'INormClass', func: BINARY_FUNC) -> 'INormClass':
    return _callable_to_norm(lambda v: func(a(v), b(v)))


def _binary_reducing(func: BINARY_FUNC, zero):

    @wraps(func)
    def _new_func(*args) -> 'INormClass':
        _sum = norm(zero)
        for item in args:
            _sum = _binary(_sum, norm(item), func)
        return _sum

    return _new_func


class INormClass:

    @abstractmethod
    def _call(self, value):
        raise NotImplementedError

    def __call__(self, value):
        return self._call(value)

    def __add__(self, other):
        return _binary(self, norm(other), operator.__add__)

    def __radd__(self, other):
        return norm(other) + self

    def __sub__(self, other):
        return _binary(self, norm(other), operator.__sub__)

    def __rsub__(self, other):
        return norm(other) - self

    def __mul__(self, other):
        return _binary(self, norm(other), operator.__mul__)

    def __rmul__(self, other):
        return norm(other) * self

    def __matmul__(self, other):
        return _binary(self, norm(other), operator.__matmul__)

    def __rmatmul__(self, other):
        return norm(other) @ self

    def __truediv__(self, other):
        return _binary(self, norm(other), operator.__truediv__)

    def __rtruediv__(self, other):
        return norm(other) / self

    def __floordiv__(self, other):
        return _binary(self, norm(other), operator.__floordiv__)

    def __rfloordiv__(self, other):
        return norm(other) // self

    def __mod__(self, other):
        return _binary(self, norm(other), operator.__mod__)

    def __rmod__(self, other):
        return norm(other) % self

    def __pow__(self, power, modulo=None):
        return _binary(self, norm(power), operator.__pow__)

    def __rpow__(self, other):
        return norm(other) ** self

    def __lshift__(self, other):
        return _binary(self, norm(other), operator.__lshift__)

    def __rlshift__(self, other):
        return norm(other) << self

    def __rshift__(self, other):
        return _binary(self, norm(other), operator.__rshift__)

    def __rrshift__(self, other):
        return norm(other) >> self

    def __and__(self, other):
        return _binary(self, norm(other), operator.__and__)

    def __rand__(self, other):
        return norm(other) & self

    def __or__(self, other):
        return _binary(self, norm(other), operator.__or__)

    def __ror__(self, other):
        return norm(other) | self

    def __xor__(self, other):
        return _binary(self, norm(other), operator.__xor__)

    def __rxor__(self, other):
        return norm(other) ^ self

    def __invert__(self):
        return _unary(self, operator.__invert__)

    def __pos__(self):
        return _unary(self, operator.__pos__)

    def __neg__(self):
        return _unary(self, operator.__neg__)

    # Attention: DO NOT USE LINKING COMPARE OPERATORS, IT WILL CAUSE ERROR.
    def __eq__(self, other):
        return _binary(self, norm(other), operator.__eq__)

    def __ne__(self, other):
        return _binary(self, norm(other), operator.__ne__)

    def __lt__(self, other):
        return _binary(self, norm(other), operator.__lt__)

    def __le__(self, other):
        return _binary(self, norm(other), operator.__le__)

    def __gt__(self, other):
        return _binary(self, norm(other), operator.__gt__)

    def __ge__(self, other):
        return _binary(self, norm(other), operator.__ge__)


lnot = normfunc(lambda x: not x)
land = _binary_reducing(lambda x, y: x and y, True)
lor = _binary_reducing(lambda x, y: x or y, True)

lin = normfunc(operator.__contains__)
lis = normfunc(operator.is_)
lisnot = normfunc(operator.is_not)

lsum = _binary_reducing(lambda x, y: x + y, 0)

_COMPARE_OPERATORS = {
    '!=': operator.__ne__,
    '==': operator.__eq__,
    '<': operator.__lt__,
    '<=': operator.__le__,
    '>': operator.__gt__,
    '>=': operator.__ge__,
}


@normfunc
def lcmp(first, *items):
    if len(items) % 2 == 1:
        raise ValueError('Count of items should be odd number but {number} found.'.format(number=len(items) + 1))

    ops, items = items[0::2], items[1::2]
    for op in ops:
        if op not in _COMPARE_OPERATORS.keys():
            raise KeyError('Invalid compare operator - {op}.'.format(op=repr(op)))

    _last = first
    for op, item in zip(ops, items):
        if not _COMPARE_OPERATORS[op](_last, item):
            return False
        _last = item

    return True
