import pytest

from ding.utils.loader import Loader


@pytest.mark.unittest
class TestConfigLoaderBase:

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

    def test_tuple_2(self):
        _loader = Loader((lambda x: x > 0, 'value error'))
        assert _loader(1) == 1
        with pytest.raises(ValueError):
            assert _loader(0)

        _loader = Loader((lambda x: x > 0, RuntimeError('runtime error')))
        assert _loader(1) == 1
        with pytest.raises(RuntimeError):
            assert _loader(0)

        _loader = Loader((lambda x: x > 0, lambda x: SystemError('system error, value is {v}'.format(v=repr(x)))))
        assert _loader(1) == 1
        with pytest.raises(SystemError):
            assert _loader(0)

    def test_tuple_3(self):
        _loader = Loader((lambda x: x > 0, lambda x: x + 1, 'value error'))
        assert _loader(1) == 2
        assert _loader(0.5) == 1.5
        with pytest.raises(ValueError):
            assert _loader(0)

        _loader = Loader((lambda x: x > 0, lambda x: -x, RuntimeError('runtime error')))
        assert _loader(1) == -1
        assert _loader(0.5) == -0.5
        with pytest.raises(RuntimeError):
            assert _loader(0)

        _loader = Loader(
            (lambda x: x > 0, lambda x: x ** 2, lambda x: SystemError('system error, value is {v}'.format(v=repr(x))))
        )
        assert _loader(1) == 1
        assert _loader(0.5) == 0.25
        with pytest.raises(SystemError):
            assert _loader(0)

    def test_tuple_invalid(self):
        with pytest.raises(ValueError):
            Loader(())
        with pytest.raises(ValueError):
            Loader((lambda x: x > 0, ))
        with pytest.raises(ValueError):
            Loader((lambda x: x > 0, lambda x: x + 1, 'value error', None))

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

    def test_raw_loader(self):
        _loader = Loader([1, 2, 3])
        assert _loader(None) == [1, 2, 3]
