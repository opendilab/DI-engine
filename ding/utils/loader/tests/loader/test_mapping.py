import pytest

from ding.utils.loader import mapping, MappingError, mpfilter, mpkeys, mpvalues, mpitems, item, item_or, is_type, \
    optional


@pytest.mark.unittest
class TestConfigLoaderMapping:

    def test_mapping(self):
        _loader = mapping(str, optional(is_type(int) | float))
        assert _loader({'sdfjk': 1}) == {'sdfjk': 1}
        assert _loader({'a': 1, 'b': 2.4, 'c': None}) == {'a': 1, 'b': 2.4, 'c': None}
        with pytest.raises(MappingError) as ei:
            _loader({'a': 1, 345: 'sdjfhk', 'b': [], None: 389450})
        err = ei.value
        assert len(err.key_errors()) == 2
        assert len(err.value_errors()) == 2
        assert len(err.errors()) == 4
        assert {key for key, _ in err.key_errors()} == {345, None}
        assert {key for key, _ in err.value_errors()} == {345, 'b'}

        with pytest.raises(TypeError):
            _loader(1)
        with pytest.raises(TypeError):
            _loader([])

    def test_mpfilter(self):
        _loader = mpfilter(lambda k, v: k in {'a', 'b', 'sum'})
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {'a': 1, 'b': 2, 'sum': 3}

    def test_mpkeys(self):
        _loader = mpkeys()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {'a', 'b', 'sum', 'sdk'}

    def test_mpvalues(self):
        _loader = mpvalues()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {1, 2, 3, 4}

    def test_mpitems(self):
        _loader = mpitems()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {('a', 1), ('b', 2), ('sum', 3), ('sdk', 4)}

    def test_item(self):
        _loader = item('a') | item('b')
        assert _loader({'a': 1}) == 1
        assert _loader({'b': 2}) == 2
        assert _loader({'a': 3, 'b': -2}) == 3

    def test_item_or(self):
        _loader = item_or('a', 0)
        assert _loader({'a': 1}) == 1
        assert _loader({'b': 2}) == 0
