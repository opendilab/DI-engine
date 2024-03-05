import functools
import textwrap
import warnings


def deprecated(since, removed_in, up_to):
    """Decorate a function to signify its deprecation.

    Args:
        since: The version when the function was first deprecated.
        removed_in: The version when the function will be removed.
        up_to: The new API users should use.
    """

    def decorator(func):
        existing_docstring = func.__doc__ or ""

        deprecated_doc = f'.. deprecated:: {since}\n    This will be removed in {removed_in}'

        if up_to is not None:
            deprecated_doc += f', please use `{up_to}` instead.'
        else:
            deprecated_doc += '.'

        # split docstring at first occurrence of newline
        summary_and_body = existing_docstring.split("\n", 1)

        if len(summary_and_body) > 1:
            summary, body = summary_and_body

            body = textwrap.dedent(body)

            new_docstring_parts = [deprecated_doc, "\n\n", summary, body]
        else:
            summary = summary_and_body[0]

            new_docstring_parts = [deprecated_doc, "\n\n", summary]

        func.__doc__ = "".join(new_docstring_parts)

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
