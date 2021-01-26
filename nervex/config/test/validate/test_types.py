import math

import pytest

from ...validate.types import number, numeric


@pytest.mark.unittest
class TestConfigValidateTypes:
    def test_number(self):
        assert number.validate(1) is None
        assert number.validate(1.0) is None
        with pytest.raises(TypeError):
            number.validate(None)
        with pytest.raises(TypeError):
            number.validate('string')

        assert number(1)
        assert number(1.0)
        assert not number(None)
        assert not number('string')

    def test_enum(self):
        pass

    # noinspection DuplicatedCode
    def test_numeric_plain(self):
        _validator = numeric()

        assert _validator(1)
        assert _validator(1.0)
        assert _validator('1')
        assert _validator('-1.0')
        assert _validator(math.inf)
        assert _validator('inf')
        assert _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_int_ban(self):
        _validator = numeric(int_ok=False)
        assert not _validator(1)
        assert _validator(1.0)
        assert _validator('1')
        assert _validator('-1.0')
        assert _validator(math.inf)
        assert _validator('inf')
        assert _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_float_ban(self):
        _validator = numeric(float_ok=False)
        assert _validator(1)
        assert not _validator(1.0)
        assert _validator('1')
        assert not _validator('-1.0')
        assert not _validator(math.inf)
        assert not _validator('inf')
        assert not _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')

    def test_numeric_double_ban(self):
        with pytest.raises(ValueError):
            numeric(int_ok=False, float_ok=False)

    # noinspection DuplicatedCode
    def test_numeric_inf_ban(self):
        _validator = numeric(inf_ok=False)
        assert _validator(1)
        assert _validator(1.0)
        assert _validator('1')
        assert _validator('-1.0')
        assert not _validator(math.inf)
        assert not _validator('inf')
        assert not _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_str_ban(self):
        _validator = numeric(str_ok=False)
        assert _validator(1)
        assert _validator(1.0)
        assert not _validator('1')
        assert not _validator('-1.0')
        assert _validator(math.inf)
        assert not _validator('inf')
        assert not _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_inf_and_str_ban(self):
        _validator = numeric(str_ok=False, inf_ok=False)
        assert _validator(1)
        assert _validator(1.0)
        assert not _validator('1')
        assert not _validator('-1.0')
        assert not _validator(math.inf)
        assert not _validator('inf')
        assert not _validator('-inf')
        assert not _validator(math.nan)
        assert not _validator('nan')
        assert not _validator(None)
        assert not _validator('styring')
        assert not _validator('-abcdef12345')
        assert not _validator('i n  f')
