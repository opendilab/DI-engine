import pytest

from ding.utils.loader import Loader, interval, to_type, is_type, keep, optional, check_only, raw, check


@pytest.mark.unittest
class TestConfigLoaderUtils:

    def test_keep(self):
        _loader = keep()
        assert _loader(1) == 1
        assert _loader(2) == 2
        assert _loader(None) is None

    def test_raw(self):
        _loader = raw(233)
        assert _loader(1) == 233
        assert _loader(2) == 233

    def test_optional(self):
        _loader = optional(Loader(int) | float)
        assert _loader(1) == 1
        assert _loader(2.0) == 2.0
        assert _loader(None) is None
        with pytest.raises(TypeError):
            _loader('string')

    def test_check_only(self):
        tonumber = to_type(int) | to_type(float)
        _loader = tonumber >> (((lambda x: x + 1) >> interval(1, 2)) | ((lambda x: x - 1) >> interval(-2, -1)))

    def test_check(self):
        _loader = check(is_type(int) | is_type(float))
        assert _loader(1)
        assert _loader(1.2)
        assert not _loader('sjhkj')

    def test_complex_case_1(self):
        tonumber = to_type(int) | to_type(float)
        _loader = tonumber >> check_only(
            ((lambda x: x + 1) >> interval(1, 2)) | ((lambda x: x - 1) >> interval(-2, -1))
        )
        assert _loader(1) == 1
        assert _loader(-1) == -1
        with pytest.raises(ValueError):
            _loader(2)
        with pytest.raises(ValueError):
            _loader(-2.0)
