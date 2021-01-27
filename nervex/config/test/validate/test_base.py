import pytest

from ...loader.base import Loader


@pytest.mark.unittest
class TestConfigValidateBase:

    def test_validator(self):
        assert Loader(int).load(1) is None
        with pytest.raises(TypeError):
            Loader(int).load('string')

        assert Loader(int).check(1)
        assert not Loader(int).check('string')

        assert Loader(int)(1)
        assert not Loader(int)('string')

        assert Loader(str).load('string') is None
        with pytest.raises(TypeError):
            Loader(str).load(1)

        assert Loader(str).check('string')
        assert not Loader(str).check(1)

        assert Loader(str)('string')
        assert not Loader(str)(1)

    def test_or(self):
        _validator = Loader(int) | str
        assert _validator.load(1) is None
        assert _validator.load('string') is None
        with pytest.raises(TypeError):
            _validator.load([])

        assert _validator(1)
        assert _validator('string')
        assert not _validator([])

    def test_ror(self):
        negative_validator = (lambda x: x < 0, lambda v: ValueError('negative number required'))
        _validator = negative_validator | Loader(int)

        assert _validator.load(-1) is None
        assert _validator.load(1) is None
        assert _validator.load(-1.0) is None
        with pytest.raises(TypeError):
            _validator.load(1.0)

        assert _validator(-1)
        assert _validator(1)
        assert _validator(-1.0)
        assert not _validator(1.0)

    # noinspection DuplicatedCode
    def test_and(self):
        positive_validator = (lambda x: x >= 0, lambda v: ValueError('non-negative number required'))
        _validator = Loader(int) & positive_validator

        assert _validator.load(1) is None
        with pytest.raises(TypeError):
            _validator.load(1.0)
        with pytest.raises(ValueError):
            _validator.load(-1)
        with pytest.raises(TypeError):
            _validator.load(-1.0)

        assert _validator(1)
        assert not _validator(1.0)
        assert not _validator(-1)
        assert not _validator(-1.0)

    # noinspection DuplicatedCode
    def test_rand(self):
        positive_validator = (lambda x: x >= 0, lambda v: ValueError('non-negative number required'))
        _validator = positive_validator & Loader(int)

        assert _validator.load(1) is None
        with pytest.raises(TypeError):
            _validator.load(1.0)
        with pytest.raises(ValueError):
            _validator.load(-1)
        with pytest.raises(ValueError):
            _validator.load(-1.0)

        assert _validator(1)
        assert not _validator(1.0)
        assert not _validator(-1)
        assert not _validator(-1.0)

    def test_bool(self):
        _validator = Loader(int) & True
        assert _validator.load(1) is None
        with pytest.raises(TypeError):
            _validator.load(None)

        assert _validator(1)
        assert not _validator(None)

        _validator = Loader(int) & False
        with pytest.raises(ValueError):
            _validator.load(1)
        with pytest.raises(TypeError):
            _validator.load(None)

        assert not _validator(1)
        assert not _validator(None)

        _validator = Loader(int) | True
        assert _validator.load(1) is None
        assert _validator.load(None) is None

        assert _validator(1)
        assert _validator(None)

        _validator = Loader(int) | False
        assert _validator.load(1) is None
        with pytest.raises(ValueError):
            _validator.load(None)

        assert _validator(1)
        assert not _validator(None)

    def test_none(self):
        _validator = Loader(int) | None
        assert _validator.load(1) is None
        assert _validator.load(None) is None
        with pytest.raises(TypeError):
            raise _validator.load('string')

        assert _validator(1)
        assert _validator(None)
        assert not _validator('string')

    def test_unknown_validator(self):
        with pytest.raises(TypeError):
            Loader([1, 2, 3])
