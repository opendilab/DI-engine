import pytest

from ...loader.string import enum


@pytest.mark.unittest
class TestConfigValidateTypes:

    def test_enum_plain(self):
        _loader = enum('red', 'green', 'blue', 'yellow')
        assert _loader('red') == 'red'
        assert _loader('green') == 'green'
        assert _loader('blue') == 'blue'
        assert _loader('yellow') == 'yellow'
        with pytest.raises(ValueError):
            _loader(int)
        with pytest.raises(ValueError):
            _loader('Red')
        with pytest.raises(ValueError):
            _loader('YELLOW')
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(None)

    def test_enum_case_insensitive(self):
        _loader = enum('red', 'green', 'blue', 'yellow', case_sensitive=False)
        assert _loader('red') == 'red'
        assert _loader('green') == 'green'
        assert _loader('blue') == 'blue'
        assert _loader('yellow') == 'yellow'
        with pytest.raises(ValueError):
            _loader(int)
        assert _loader('Red') == 'red'
        assert _loader('YELLOW') == 'yellow'
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(None)
