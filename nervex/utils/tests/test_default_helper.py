import pytest
import numpy as np
from collections import namedtuple

from nervex.utils.default_helper import lists_to_dicts, dicts_to_lists, squeeze, default_get, override, error_wrapper


@pytest.mark.unittest
class TestDefaultHelper():

    def test_lists_to_dicts(self):
        with pytest.raises(ValueError):
            lists_to_dicts([])
        with pytest.raises(TypeError):
            lists_to_dicts([1])
        assert lists_to_dicts([{1: 1, 10: 3}, {1: 2, 10: 4}]) == {1: [1, 2], 10: [3, 4]}
        T = namedtuple('T', ['location', 'race'])
        data = [T({'x': 1, 'y': 2}, 'zerg') for _ in range(3)]
        output = lists_to_dicts(data)
        assert isinstance(output, T) and output.__class__ == T
        assert len(output.location) == 3

    def test_dicts_to_lists(self):
        assert dicts_to_lists({1: [1, 2], 10: [3, 4]}) == [{1: 1, 10: 3}, {1: 2, 10: 4}]

    def test_squeeze(self):
        assert squeeze((4, )) == 4
        assert squeeze({'a': 4}) == 4
        data = np.random.randn(3)
        output = squeeze(data)
        assert (output == data).all()

    def test_default_get(self):
        assert default_get({}, 'a', default_value=1, judge_fn=lambda x: x < 2) == 1
        assert default_get({}, 'a', default_fn=lambda: 1, judge_fn=lambda x: x < 2) == 1
        with pytest.raises(AssertionError):
            default_get({}, 'a', default_fn=lambda: 1, judge_fn=lambda x: x < 0)
        assert default_get({'val': 1}, 'val', default_value=2) == 1

    def test_override(self):

        class foo(object):

            def fun(self):
                raise NotImplementedError

        class foo1(foo):

            @override(foo)
            def fun(self):
                return "a"

        with pytest.raises(NameError):

            class foo2(foo):

                @override(foo)
                def func(self):
                    pass

        with pytest.raises(NotImplementedError):
            foo().fun()
        foo1().fun()

    def test_error_wrapper(self):

        def good_ret(a, b=1):
            return a + b

        wrap_good_ret = error_wrapper(good_ret, 0)
        assert good_ret(1) == wrap_good_ret(1)

        def bad_ret(a, b=0):
            return a / b

        wrap_bad_ret = error_wrapper(bad_ret, 0)
        assert wrap_bad_ret(1) == 0
