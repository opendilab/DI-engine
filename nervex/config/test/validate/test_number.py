import pytest

from ...validate.number import interval


@pytest.mark.unittest
class TestConfigValidateNumber:
    def test_interval_common(self):
        _validator = interval(1, 3.5)
        assert not _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.5)
        assert not _validator(4.0)
