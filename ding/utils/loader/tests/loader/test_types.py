import pytest
from easydict import EasyDict

from ding.utils.loader import interval, negative, is_type, to_type, prop, method, fcall, is_callable, fpartial, keep


@pytest.mark.unittest
class TestConfigLoaderTypes:

    def test_is_type(self):
        _loader = is_type(float) | is_type(int)
        assert _loader(1) == 1
        assert _loader(2.5) == 2.5
        with pytest.raises(TypeError):
            _loader(None)

    # noinspection PyTypeChecker
    def test_is_type_invalid(self):
        with pytest.raises(TypeError):
            is_type(lambda x: x + 1)

    def test_to_type_float(self):
        _loader = keep() >> to_type(float)
        assert _loader(1) == 1.0
        assert isinstance(_loader(1), float)
        assert _loader(2.0) == 2.0
        assert isinstance(_loader(2.0), float)

    def test_to_type_str(self):
        _loader = keep() >> to_type(str)
        assert _loader(1) == '1'
        assert _loader(2.0) == '2.0'
        assert _loader(None) == 'None'

    def test_to_type_float_str(self):
        _loader = keep() >> to_type(float) >> to_type(str)
        assert _loader(1) == '1.0'
        assert _loader(2.0) == '2.0'
        with pytest.raises(TypeError):
            _loader(None)

    def test_is_callable(self):
        _loader = is_callable()
        assert _loader(lambda x: 1)
        assert _loader(str) == str
        assert _loader(str.lower) == str.lower
        with pytest.raises(TypeError):
            _loader(1)

    def test_prop(self):
        t1 = EasyDict({'x': 1, 'y': 2, 'z': 'string'})
        t2 = EasyDict({'x': 'str'})
        t3 = EasyDict({'z': -1, 'y': 'sss'})

        _loader = prop('x') >> int
        assert _loader(t1) == 1
        with pytest.raises(TypeError):
            _loader(t2)
        with pytest.raises(AttributeError):
            _loader(t3)

        _loader = (prop('x') >> str) | (prop('y') >> str) | (prop('z') >> str)
        assert _loader(t1) == 'string'
        assert _loader(t2) == 'str'
        assert _loader(t3) == 'sss'

    def test_method(self):
        t1 = 'STRING'
        t2 = 2
        t3 = EasyDict({'lower': 1})

        _loader = method('lower')
        assert _loader(t1)() == 'string'
        with pytest.raises(TypeError):
            _loader(t2)
        with pytest.raises(TypeError):
            _loader(t3)

    def test_fcall(self):
        _loader = fcall('STRING')
        assert _loader(lambda x: len(x)) == 6
        assert _loader(str.lower) == 'string'

    def test_fpartial(self):
        _loader = fpartial(x=2)

        def _func_1(x, y):
            return x + y

        def _func_2(x, y):
            return x * y

        assert _loader(_func_1)(y=6) == 8
        assert _loader(_func_2)(y=6) == 12

    def test_func_complex_case_1(self):
        _loader = fpartial(x=1) >> ((fcall(y=1) >> interval(0, None)) | (fcall(y=2) >> interval(None, 0) >> negative()))

        def _func_1(x, y):
            return x + y

        def _func_2(x, y):
            return 5 * x - 4 * y

        def _func_3(x, y):
            return -5 * x - 4 * y

        assert _loader(_func_1) == 2
        assert _loader(_func_2) == 1
        assert _loader(_func_3) == 13
