import pytest

from ...loader.base import Loader
from ...loader.collection import collection, contains, length_is, length


@pytest.mark.unittest
class TestConfigValidateCollection:

    def test_collection(self):
        _validator = collection(Loader(int) | str)
        assert _validator([1]) == [1]
        assert _validator([1, 'string']) == [1, 'string']
        assert _validator({1, 'string'}) == {1, 'string'}
        assert _validator((1, 'string')) == (1, 'string')
        assert _validator({'string': 1, 'new_string': 2}) == {'string': 1, 'new_string': 2}
        with pytest.raises(TypeError):
            _validator(1)
        with pytest.raises(TypeError):
            _validator(None)
        with pytest.raises(TypeError):
            _validator([None, 1, 'string'])

    def test_length_min_length(self):
        _validator = length(min_length=2)
        assert _validator('ab')
        assert _validator('abcdefg')
        assert _validator([1, 2])
        assert _validator([1, 2, 3, 4, 5, 6, 7])
        assert not _validator('a')
        assert not _validator([1])
        assert _validator('abcdefghij')
        assert _validator([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    def test_length_max_length(self):
        _validator = length(max_length=7)
        assert _validator('ab')
        assert _validator('abcdefg')
        assert _validator([1, 2])
        assert _validator([1, 2, 3, 4, 5, 6, 7])
        assert _validator('a')
        assert _validator([1])
        assert not _validator('abcdefghij')
        assert not _validator([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    def test_length_both_length(self):
        _validator = length(min_length=2, max_length=7)
        assert _validator('ab')
        assert _validator('abcdefg')
        assert _validator([1, 2])
        assert _validator([1, 2, 3, 4, 5, 6, 7])
        assert not _validator('a')
        assert not _validator([1])
        assert not _validator('abcdefghij')
        assert not _validator([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    def test_length_is(self):
        _validator = length_is(10)
        assert _validator('abcdefghij')
        assert _validator([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        assert not _validator('abcdefg')
        assert not _validator('abcdefghijk')
        assert not _validator([1, 2, 3, 4])
        assert not _validator([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
        assert not _validator(1)

    def test_contains(self):
        _validator = contains('item') & list & collection(str)
        assert _validator(['item'])
        assert _validator(['item', 'string_1', 'string_2'])
        assert not _validator(('item',))
        assert not _validator(('item', 'string_1', 'string_2'))
        assert not _validator(['item', 1])
        assert not _validator(['itemx'])
        assert not _validator(1)
