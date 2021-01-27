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

    def test_interval_all(self):
        _validator = interval()
        assert _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.5)
        assert _validator(4.0)

    def test_interval_left_open(self):
        _validator = interval(1.0, left_ok=False)
        assert not _validator(0.5)
        assert not _validator(0.9)
        assert not _validator(0.999)
        assert not _validator(1.0)
        assert _validator(1.001)
        assert _validator(1.1)
        assert _validator(1.5)
        assert _validator(3.5)
        assert _validator(4.0)

    def test_interval_left_open_eps(self):
        _validator = interval(1.0, left_ok=False, eps=0.01)
        assert not _validator(0.5)
        assert not _validator(0.9)
        assert not _validator(0.999)
        assert not _validator(1.0)
        assert not _validator(1.001)
        assert _validator(1.1)
        assert _validator(1.5)
        assert _validator(3.5)
        assert _validator(4.0)

    def test_interval_left_close(self):
        _validator = interval(1.0)
        assert not _validator(0.5)
        assert not _validator(0.9)
        assert not _validator(0.999)
        assert _validator(1.0)
        assert _validator(1.001)
        assert _validator(1.1)
        assert _validator(1.5)
        assert _validator(3.5)
        assert _validator(4.0)

    def test_interval_left_close_eps(self):
        _validator = interval(1.0, eps=0.01)
        assert not _validator(0.5)
        assert not _validator(0.9)
        assert _validator(0.999)
        assert _validator(1.0)
        assert _validator(1.001)
        assert _validator(1.1)
        assert _validator(1.5)
        assert _validator(3.5)
        assert _validator(4.0)

    def test_interval_right_open(self):
        _validator = interval(right=3.5, right_ok=False)
        assert _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.4)
        assert _validator(3.499)
        assert not _validator(3.5)
        assert not _validator(3.501)
        assert not _validator(3.6)
        assert not _validator(4.0)

    def test_interval_right_open_eps(self):
        _validator = interval(right=3.5, right_ok=False, eps=0.01)
        assert _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.4)
        assert not _validator(3.499)
        assert not _validator(3.5)
        assert not _validator(3.501)
        assert not _validator(3.6)
        assert not _validator(4.0)

    def test_interval_right_close(self):
        _validator = interval(right=3.5)
        assert _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.4)
        assert _validator(3.499)
        assert _validator(3.5)
        assert not _validator(3.501)
        assert not _validator(3.6)
        assert not _validator(4.0)

    def test_interval_right_close_eps(self):
        _validator = interval(right=3.5, eps=0.01)
        assert _validator(0.5)
        assert _validator(1.0)
        assert _validator(1.5)
        assert _validator(3.4)
        assert _validator(3.499)
        assert _validator(3.5)
        assert _validator(3.501)
        assert not _validator(3.6)
        assert not _validator(4.0)

    def test_interval_invalid(self):
        with pytest.raises(ValueError):
            interval(1.0, 0.9)
