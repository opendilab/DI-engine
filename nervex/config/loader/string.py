from functools import wraps
from typing import Callable

from .base import Loader, ILoaderClass

STRING_PROCESSOR = Callable[[str], str]


def enum(*items, case_sensitive: bool = True) -> ILoaderClass:
    def _case_sensitive(func: STRING_PROCESSOR) -> STRING_PROCESSOR:
        if case_sensitive:
            return func
        else:
            @wraps(func)
            def _new_func(value: str) -> str:
                return func(value).lower()

            return _new_func

    @_case_sensitive
    def _item_process(value):
        return str(value)

    item_set = set([_item_process(item) for item in items])

    def _load(value: str):
        real_value = _item_process(value)
        if real_value not in item_set:
            raise ValueError('unknown enum value {value}'.format(value=repr(value)))

        return real_value

    return Loader(_load)
