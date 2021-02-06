import re
from functools import wraps
from itertools import islice
from typing import Callable, Union, Pattern

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


def _to_regexp(regexp) -> Pattern:
    if isinstance(regexp, Pattern):
        return regexp
    elif isinstance(regexp, str):
        return re.compile(regexp)
    else:
        raise TypeError(
            'Regexp should be either str or re.Pattern but {actual} found.'.format(actual=repr(type(regexp).__name__))
        )


def rematch(regexp: Union[str, Pattern]) -> ILoaderClass:
    regexp = _to_regexp(regexp)

    def _load(value: str):
        if not regexp.fullmatch(value):
            raise ValueError(
                'fully match with regexp {pattern} expected but {actual} found'.format(
                    pattern=repr(regexp.pattern),
                    actual=repr(value),
                )
            )

        return value

    return Loader(_load)


def regrep(regexp: Union[str, Pattern], group: int = 0) -> ILoaderClass:
    regexp = _to_regexp(regexp)

    def _load(value: str):
        results = list(islice(regexp.finditer(value), 1))
        if results:
            return results[0][group]
        else:
            raise ValueError(
                'fully match with regexp {pattern} expected but {actual} found'.format(
                    pattern=repr(regexp.pattern),
                    actual=repr(value),
                )
            )

    return Loader(_load)
