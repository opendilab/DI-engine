from typing import Iterable, TypeVar, Callable

_IterType = TypeVar('_IterType')
_IterTargetType = TypeVar('_IterTargetType')


def iter_mapping(iter_: Iterable[_IterType], mapping: Callable[[_IterType], _IterTargetType]):
    r"""
    Overview:
        Map a list of iterable elements to input iteration callable
    Arguments:
        - iter_(:obj:`_IterType list`): The list for iteration
        - mapping (:obj:`Callable [[_IterType], _IterTargetType]`): A callable that maps iterable elements function.
    Return:
        - (:obj:`iter_mapping object`): Iteration results
    Example:
        >>> iterable_list = [1, 2, 3, 4, 5]
        >>> _iter = iter_mapping(iterable_list, lambda x: x ** 2)
        >>> print(list(_iter))
        [1, 4, 9, 16, 25]
    """
    for item in iter_:
        yield mapping(item)
