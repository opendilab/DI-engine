import pytest

from ...loader.base import Loader


@pytest.mark.unittest
class TestConfigValidateBase:

    def test_load(self):
        _loader = Loader(int)
        assert _loader.load(1) == 1
        with pytest.raises(TypeError):
            _loader.load('string')

    def test_check(self):
        _loader = Loader(int)
        assert _loader.check(1)
        assert not _loader.check('string')

    def test_call(self):
        _loader = Loader(int)
        assert _loader(1) == 1
        with pytest.raises(TypeError):
            _loader('string')

    def test_or(self):
        _loader = Loader(int) | str
        assert _loader(1) == 1
        assert _loader('string') == 'string'
        with pytest.raises(TypeError):
            _loader([])

        assert _loader.check(1)
        assert _loader.check('string')
        assert not _loader.check([])

    def test_ror(self):
        _loader = (lambda v: v < 0, 'Negative number expected.') | Loader(int)

        assert _loader(-1) == -1
        assert _loader(1) == 1
        assert _loader(-1.0) - 1.0
        with pytest.raises(TypeError):
            _loader(1.0)

        assert _loader.check(-1)
        assert _loader.check(1)
        assert _loader.check(-1.0)
        assert not _loader.check(1.0)

    # noinspection DuplicatedCode
    def test_and(self):
        _loader = Loader(int) & (lambda x: x >= 0, 'non-negative number required')

        assert _loader(1) == 1
        with pytest.raises(TypeError):
            _loader(1.0)
        with pytest.raises(ValueError):
            _loader(-1)
        with pytest.raises(TypeError):
            _loader(-1.0)

        assert _loader.check(1)
        assert not _loader.check(1.0)
        assert not _loader.check(-1)
        assert not _loader.check(-1.0)

    # noinspection DuplicatedCode
    def test_rand(self):
        _loader = (lambda x: x >= 0, 'non-negative number required') & Loader(int)

        assert _loader(1) == 1
        with pytest.raises(TypeError):
            _loader(1.0)
        with pytest.raises(ValueError):
            _loader(-1)
        with pytest.raises(ValueError):
            _loader(-1.0)

        assert _loader.check(1)
        assert not _loader.check(1.0)
        assert not _loader.check(-1)
        assert not _loader.check(-1.0)

    def test_bool(self):
        _loader = Loader(int) & True
        assert _loader(1) == 1
        with pytest.raises(TypeError):
            _loader(None)

        assert _loader.check(1)
        assert not _loader.check(None)

        _loader = Loader(int) & False
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(TypeError):
            _loader(None)

        assert not _loader.check(1)
        assert not _loader.check(None)

        _loader = Loader(int) | True
        assert _loader(1) == 1
        assert _loader(None) is None

        assert _loader.check(1)
        assert _loader.check(None)

        _loader = Loader(int) | False
        assert _loader(1) == 1
        with pytest.raises(ValueError):
            _loader(None)

        assert _loader.check(1)
        assert not _loader.check(None)

    def test_none(self):
        _loader = Loader(int) | None
        assert _loader(1) == 1
        assert _loader(None) is None
        with pytest.raises(TypeError):
            _loader('string')

        assert _loader.check(1)
        assert _loader.check(None)
        assert not _loader.check('string')

    def test_unknown_loader(self):
        with pytest.raises(TypeError):
            Loader([1, 2, 3])
