import pytest

from ding.utils.loader import Loader, collection, contains, length_is, length, tuple_, CollectionError, cofilter, \
    tpselector, plus, minus, interval, negative, to_type, optional, check_only


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
        with pytest.raises(CollectionError) as ei:
            _loader([None, 1, 'string', 290384.23])

        err = ei.value
        assert len(err.errors) == 2
        assert [index for index, _ in err.errors] == [0, 3]
        assert [type(item) for _, item in err.errors] == [TypeError, TypeError]

    def test_collection_map(self):
        _loader = collection(
            ((Loader(int) | float) >> plus(1) >> negative()) | (str >> (to_type(int) | to_type(float)))
        )
        assert _loader([1, 2, -3.0, '1', '2.0']) == [-2, -3, 2.0, 1, 2.0]
        assert [type(item) for item in _loader([1, 2, -3.0, '1', '2.0'])] == [int, int, float, int, float]

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
        with pytest.raises(CollectionError) as ei:
            _loader(['item', 1, [1, 2]])
        err = ei.value
        assert len(err.errors) == 2
        assert [index for index, _ in err.errors] == [1, 2]
        assert [type(item) for _, item in err.errors] == [TypeError, TypeError]

        with pytest.raises(ValueError):
            _loader(['itemx'])
        with pytest.raises(TypeError):
            _loader(1)

    def test_cofilter(self):
        _loader = cofilter(lambda x: x > 0)

        assert _loader([1, 2, -1, 3, -2]) == [1, 2, 3]
        assert _loader((1, 2, -1, 3, -2)) == (1, 2, 3)
        assert _loader({1, 2, -1, 3, -2}) == {1, 2, 3}

    def test_cofilter_complex_case_1(self):
        _loader = list & check_only(
            (cofilter(lambda x: x == 1) >> length_is(5)) & (cofilter(lambda x: x == 0) >> length_is(5))
        )

        assert _loader([1, 0, 0, 1, 0, 1, 0, 0, 1, 1]) == [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
        assert _loader([1, 2, -3, 0, 0, 1, 0, 1, 0, 0, 1, 1]) == [1, 2, -3, 0, 0, 1, 0, 1, 0, 0, 1, 1]
        with pytest.raises(ValueError):
            _loader([1, 0, 0, 1, 0, 1, 0, 0, 1, -1])

    def test_tpselector(self):
        _loader = tpselector(0, 2)

        assert _loader((1, 2, 3)) == (1, 3)
        assert _loader((int, None, {}, 4)) == (int, {})
