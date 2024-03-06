import functools
import textwrap
import warnings
from typing import Optional


def deprecated(since: str, removed_in: str, up_to: Optional[str] = None):
    """
    Overview:
        Decorate a function to signify its deprecation.
    Arguments:
        - since (:obj:`str`): the version when the function was first deprecated.
        - removed_in (:obj:`str`): the version when the function will be removed.
        - up_to (:obj:`Optional[str]`): the new API users should use.
    Returns:
        - decorator (:obj:`Callable`): decorated function.
    Examples:
        >>> from ding.utils.deprecation import deprecated
        >>> @deprecated('0.4.1', '0.5.1')
        >>> def hello():
        >>>     print('hello')
    """

    def decorator(func):
        existing_docstring = func.__doc__ or ""

        deprecated_doc = f'.. deprecated:: {since}\n    Deprecated and will be removed in version {removed_in}'

        if up_to is not None:
            deprecated_doc += f', please use `{up_to}` instead.'
        else:
            deprecated_doc += '.'

        func.__doc__ = deprecated_doc + "\n\n" + textwrap.dedent(existing_docstring)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = (
                f'API `{func.__module__}.{func.__name__}` is deprecated since version {since} '
                f'and will be removed in version {removed_in}'
            )
            if up_to is not None:
                warning_msg += f", please use `{up_to}` instead."
            else:
                warning_msg += "."

            warnings.warn(warning_msg, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
