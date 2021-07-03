import pytest

from ding.utils.loader import dict_, DictError, item, norm, msum, keep


@pytest.mark.unittest
class TestConfigLoaderDict:

    def test_dict(self):
        _loader = dict_(b=item('a'), a=item('b'))
        assert _loader({'a': 1, 'b': 2}) == {'a': 2, 'b': 1}
        assert _loader({'a': 4, 'b': [1, 2]}) == {'a': [1, 2], 'b': 4}

        with pytest.raises(DictError) as ei:
            _loader({'a': 1, 'bb': 2})
        err = ei.value
        assert set(err.errors.keys()) == {'a'}
        assert isinstance(err.errors['a'], KeyError)

    def test_dict_complex_case_1(self):
        _loader = dict_(
            real=msum(item('a'), item('b')),
            result=item('sum') | item('result'),
        ) >> dict_(
            real=item('real') >> keep(),
            result=item('result') >> keep(),
            correct=norm(item('real')) == norm(item('result')),
        )
        assert _loader({'a': 1, 'b': 2, 'result': 3}) == {'real': 3, 'result': 3, 'correct': True}
        assert _loader({'a': 2, 'b': 2, 'sum': 3}) == {'real': 4, 'result': 3, 'correct': False}
