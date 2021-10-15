import os
import sys
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import Callable, Any

_global_no_output_lock = Lock()


@contextmanager
def silence(no_stdout: bool = True, no_stderr: bool = True):
    with _global_no_output_lock:
        if no_stdout:
            # Don't use `wb` mode here, otherwise it will cause all streaming methods to crash
            _real_stdout, sys.stdout = sys.stdout, open(os.devnull, 'w')
        if no_stderr:
            _real_stderr, sys.stderr = sys.stderr, open(os.devnull, 'w')

        try:
            yield
        finally:
            if no_stdout:
                sys.stdout = _real_stdout
            if no_stderr:
                sys.stderr = _real_stderr


def silence_function(no_stdout: bool = True, no_stderr: bool = True):

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(func)
        def _func(*args, **kwargs):
            with silence(no_stdout, no_stderr):
                return func(*args, **kwargs)

        return _func

    return _decorator
