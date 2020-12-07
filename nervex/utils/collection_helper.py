from typing import Iterable, TypeVar, Callable

_IterType = TypeVar('_IterType')
_IterTargetType = TypeVar('_IterTargetType')


def iter_mapping(iter_: Iterable[_IterType], mapping: Callable[[_IterType], _IterTargetType]):
    for item in iter_:
        yield mapping(item)
