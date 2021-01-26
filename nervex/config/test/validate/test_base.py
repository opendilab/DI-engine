import pytest

from ...validate.base import Validator


@pytest.mark.unittest
class TestConfigValidateBase:

    def test_validator(self):
        assert Validator(int).validate(1) is None
        with pytest.raises(TypeError):
            Validator(int).validate('string')

        assert Validator(int).check(1)
        assert not Validator(int).check('string')

        assert Validator(int)(1)
        assert not Validator(int)('string')

        assert Validator(str).validate('string') is None
        with pytest.raises(TypeError):
            Validator(str).validate(1)

        assert Validator(str).check('string')
        assert not Validator(str).check(1)

        assert Validator(str)('string')
        assert not Validator(str)(1)

    def test_or(self):
        _validator = Validator(int) | str
        assert _validator.validate(1) is None
        assert _validator.validate('string') is None
        with pytest.raises(TypeError):
            _validator.validate([])

        assert _validator(1)
        assert _validator('string')
        assert not _validator([])

    def test_ror(self):
        negative_validator = (lambda x: x < 0, lambda v: ValueError('negative number required'))
        _validator = negative_validator | Validator(int)

        assert _validator.validate(-1) is None
        assert _validator.validate(1) is None
        assert _validator.validate(-1.0) is None
        with pytest.raises(TypeError):
            _validator.validate(1.0)

        assert _validator(-1)
        assert _validator(1)
        assert _validator(-1.0)
        assert not _validator(1.0)

    # noinspection DuplicatedCode
    def test_and(self):
        positive_validator = (lambda x: x >= 0, lambda v: ValueError('non-negative number required'))
        _validator = Validator(int) & positive_validator

        assert _validator.validate(1) is None
        with pytest.raises(TypeError):
            _validator.validate(1.0)
        with pytest.raises(ValueError):
            _validator.validate(-1)
        with pytest.raises(TypeError):
            _validator.validate(-1.0)

        assert _validator(1)
        assert not _validator(1.0)
        assert not _validator(-1)
        assert not _validator(-1.0)

    # noinspection DuplicatedCode
    def test_rand(self):
        positive_validator = (lambda x: x >= 0, lambda v: ValueError('non-negative number required'))
        _validator = positive_validator & Validator(int)

        assert _validator.validate(1) is None
        with pytest.raises(TypeError):
            _validator.validate(1.0)
        with pytest.raises(ValueError):
            _validator.validate(-1)
        with pytest.raises(ValueError):
            _validator.validate(-1.0)

        assert _validator(1)
        assert not _validator(1.0)
        assert not _validator(-1)
        assert not _validator(-1.0)

    def test_bool(self):
        _validator = Validator(int) & True
        assert _validator.validate(1) is None
        with pytest.raises(TypeError):
            _validator.validate(None)

        assert _validator(1)
        assert not _validator(None)

        _validator = Validator(int) & False
        with pytest.raises(ValueError):
            _validator.validate(1)
        with pytest.raises(TypeError):
            _validator.validate(None)

        assert not _validator(1)
        assert not _validator(None)

        _validator = Validator(int) | True
        assert _validator.validate(1) is None
        assert _validator.validate(None) is None

        assert _validator(1)
        assert _validator(None)

        _validator = Validator(int) | False
        assert _validator.validate(1) is None
        with pytest.raises(ValueError):
            _validator.validate(None)

        assert _validator(1)
        assert not _validator(None)

    def test_none(self):
        _validator = Validator(int) | None
        assert _validator.validate(1) is None
        assert _validator.validate(None) is None
        with pytest.raises(TypeError):
            raise _validator.validate('string')

        assert _validator(1)
        assert _validator(None)
        assert not _validator('string')

    def test_unknown_validator(self):
        with pytest.raises(TypeError):
            Validator([1, 2, 3])
