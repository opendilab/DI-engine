import operator
from abc import abstractmethod
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


def _unary(a: 'INormClass', func: Callable[[Any], Any]) -> 'INormClass':
    return _callable_to_norm(lambda v: func(a(v)))


def _binary(a: 'INormClass', b: 'INormClass', func: Callable[[Any, Any], Any]) -> 'INormClass':
    return _callable_to_norm(lambda v: func(a(v), b(v)))


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

    def __abs__(self):
        return _unary(self, operator.__abs__)

    def __int__(self):
        return _unary(self, lambda x: int(x))

    def __float__(self):
        return _unary(self, lambda x: float(x))

    def __bool__(self):
        return _unary(self, lambda x: not not x)


def lnot(a) -> INormClass:
    return _unary(norm(a), lambda x: not x)


def _reducing(func: Callable[[Any, Any], Any]):
    def _new_func(a, b, *cs):
        _result = func(a, b)
        for c in cs:
            _result = func(_result, c)
        return _result

    return _new_func


@_reducing
def land(a, b) -> INormClass:
    return _binary(norm(a), norm(b), lambda x, y: x and y)


@_reducing
def lor(a, b) -> INormClass:
    return _binary(norm(a), norm(b), lambda x, y: x or y)


def lfunc(func, *args, **kwargs) -> INormClass:
    args = [norm(item) for item in args]
    kwargs = [(norm(key), norm(value)) for key, value in kwargs.items()]

    def _call(v):
        _args = [item(v) for item in args]
        _kwargs = {key(v): value(v) for key, value in kwargs}
        return func(*_args, **_kwargs)

    return _callable_to_norm(_call)


def lin(a, b) -> INormClass:
    return _binary(norm(a), norm(b), operator.__contains__)


def lis(a, b) -> INormClass:
    return _binary(norm(a), norm(b), operator.is_)


def lisnot(a, b) -> INormClass:
    return _binary(norm(a), norm(b), operator.is_not)
