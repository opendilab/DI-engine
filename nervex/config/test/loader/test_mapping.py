import pytest

from ...loader.mapping import mapping, MappingError, mpfilter, keys, values, items
from ...loader.types import is_type
from ...loader.utils import optional


@pytest.mark.unittest
class TestConfigLoaderCollection:

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

    def test_keys(self):
        _loader = keys()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {'a', 'b', 'sum', 'sdk'}

    def test_values(self):
        _loader = values()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {1, 2, 3, 4}

    def test_items(self):
        _loader = items()
        assert _loader({'a': 1, 'b': 2, 'sum': 3, 'sdk': 4}) == {('a', 1), ('b', 2), ('sum', 3), ('sdk', 4)}
