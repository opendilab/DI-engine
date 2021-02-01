import pytest

from ...loader.dict import dict_, DictError
from ...loader.mapping import index


@pytest.mark.unittest
class TestConfigLoaderDict:

    def test_dict(self):
        _loader = dict_(b=index('a'), a=index('b'))
        assert _loader({'a': 1, 'b': 2}) == {'a': 2, 'b': 1}
        assert _loader({'a': 4, 'b': [1, 2]}) == {'a': [1, 2], 'b': 4}

        with pytest.raises(DictError) as ei:
            _loader({'a': 1, 'bb': 2})
        err = ei.value
        assert set(err.errors.keys()) == {'a'}
        assert isinstance(err.errors['a'], KeyError)
