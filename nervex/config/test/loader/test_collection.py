import pytest

from ...loader.base import Loader
from ...loader.collection import collection, contains, length_is, length, tuple_
from ...loader.number import plus, minus, interval
from ...loader.utils import optional


@pytest.mark.unittest
class TestConfigLoaderCollection:

    def test_collection(self):
        _loader = collection(Loader(int) | str)
        assert _loader([1]) == [1]
        assert _loader([1, 'string']) == [1, 'string']
        assert _loader({1, 'string'}) == {1, 'string'}
        assert _loader((1, 'string')) == (1, 'string')
        with pytest.raises(TypeError):
            _loader(1)
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader([None, 1, 'string'])

    def test_tuple(self):
        _loader = tuple_(int, optional(float), plus(1) >> interval(2, 3), minus(1) >> interval(-4, -3))
        assert _loader((1, 2.3, 1.2, -2.5)) == (1, 2.3, 2.2, -3.5)
        assert _loader((10, None, 2, -3)) == (10, None, 3, -4)
        with pytest.raises(TypeError):
            _loader((10.1, 9238.2, 1.2, -2.5))
        with pytest.raises(ValueError):
            _loader((10, 9238.2, 4.2, -2.5))

    # noinspection DuplicatedCode
    def test_length_min_length(self):
        _loader = length(min_length=2)
        assert _loader('ab') == 'ab'
        assert _loader('abcdefg') == 'abcdefg'
        assert _loader([1, 2]) == [1, 2]
        assert _loader([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]
        with pytest.raises(ValueError):
            _loader('a')
        with pytest.raises(ValueError):
            _loader([1])
        assert _loader('abcdefghij') == 'abcdefghij'
        assert _loader([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        with pytest.raises(TypeError):
            _loader(1)

    # noinspection DuplicatedCode
    def test_length_max_length(self):
        _loader = length(max_length=7)
        assert _loader('ab') == 'ab'
        assert _loader('abcdefg') == 'abcdefg'
        assert _loader([1, 2]) == [1, 2]
        assert _loader([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]
        assert _loader('a') == 'a'
        assert _loader([1]) == [1]
        with pytest.raises(ValueError):
            _loader('abcdefghij')
        with pytest.raises(ValueError):
            _loader([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        with pytest.raises(TypeError):
            _loader(1)

    # noinspection DuplicatedCode
    def test_length_both_length(self):
        _loader = length(min_length=2, max_length=7)
        assert _loader('ab') == 'ab'
        assert _loader('abcdefg') == 'abcdefg'
        assert _loader([1, 2]) == [1, 2]
        assert _loader([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5, 6, 7]
        with pytest.raises(ValueError):
            _loader('a')
        with pytest.raises(ValueError):
            _loader([1])
        with pytest.raises(ValueError):
            _loader('abcdefghij')
        with pytest.raises(ValueError):
            _loader([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        with pytest.raises(TypeError):
            _loader(1)

    def test_length_is(self):
        _loader = length_is(10)
        assert _loader('abcdefghij') == 'abcdefghij'
        assert _loader([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        with pytest.raises(ValueError):
            _loader('abcdefg')
        with pytest.raises(ValueError):
            _loader('abcdefghijk')
        with pytest.raises(ValueError):
            _loader([1, 2, 3, 4])
        with pytest.raises(ValueError):
            _loader([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
        with pytest.raises(TypeError):
            _loader(1)

    def test_contains(self):
        _loader = contains('item') & list & collection(str)
        assert _loader(['item']) == ['item']
        assert _loader(['item', 'string_1', 'string_2']) == ['item', 'string_1', 'string_2']
        with pytest.raises(TypeError):
            _loader(('item', ))
        with pytest.raises(TypeError):
            _loader(('item', 'string_1', 'string_2'))
        with pytest.raises(TypeError):
            _loader(['item', 1])
        with pytest.raises(ValueError):
            _loader(['itemx'])
        with pytest.raises(TypeError):
            _loader(1)
