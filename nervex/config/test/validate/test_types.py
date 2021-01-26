import pytest

from ...validate.types import number


@pytest.mark.unittest
class TestConfigValidateTypes:
    def test_number(self):
        assert number.validate(1) is None
        assert number.validate(1.0) is None
        with pytest.raises(TypeError):
            number.validate(None)
        with pytest.raises(TypeError):
            number.validate('string')

        assert number.check(1)
        assert number.check(1.0)
        assert not number.check(None)
        assert not number.check('string')
