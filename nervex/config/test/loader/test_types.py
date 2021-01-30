import pytest

from ...loader.types import is_type, to_type
from ...loader.utils import keep


@pytest.mark.unittest
class TestConfigLoaderTypes:

    def test_is_type(self):
        _loader = is_type(float) | is_type(int)
        assert _loader(1) == 1
        assert _loader(2.5) == 2.5
        with pytest.raises(TypeError):
            _loader(None)

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
