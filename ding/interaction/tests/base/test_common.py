import string
import time
from typing import Any, Callable

import pytest

from ...base import random_token, translate_dict_func, default_func, ControllableService


@pytest.mark.unittest
class TestInteractionBaseCommon:

    def test_random_token(self):
        assert len(random_token()) == 64
        assert len(random_token(32)) == 32
        assert set(random_token()) - set(string.hexdigits) == set()

    def test_translate_dict_func(self):
        assert translate_dict_func({
            'a': lambda: 2,
            'b': lambda: 3,
            'sum': lambda: 5,
        })() == {
            'a': 2,
            'b': 3,
            'sum': 5
        }
        assert translate_dict_func(
            {
                'a': lambda ax, bx: 2 + ax,
                'b': lambda ax, bx: 3 + bx,
                'sum': lambda ax, bx: 5 + ax + bx,
            }
        )(4, 5) == {
            'a': 6,
            'b': 8,
            'sum': 14
        }

    def test_default_func(self):

        def _calculate(a: int, b: int, callback: Callable[..., Any] = None):
            return default_func(233)(callback)(a, b)

        assert _calculate(1, 2) == 233
        assert _calculate(1, 2, lambda a, b: a + b) == 3
        assert _calculate(1, 2, lambda a, b: a * b) == 2


@pytest.mark.unittest
class TestInteractionBaseControllableService:

    def test_it(self):
        _start, _shutdown, _finished = False, False, False

        class _Service(ControllableService):

            def start(self):
                nonlocal _start
                _start = True

            def shutdown(self):
                nonlocal _shutdown
                _shutdown = True

            def join(self):
                time.sleep(1.0)
                nonlocal _finished
                _finished = True

        assert (_start, _shutdown, _finished) == (False, False, False)
        with _Service():
            assert (_start, _shutdown, _finished) == (True, False, False)

        assert (_start, _shutdown, _finished) == (True, True, True)
