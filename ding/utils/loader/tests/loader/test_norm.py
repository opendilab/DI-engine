from ditk import logging

import pytest

from ding.utils.loader import Loader, interval, item, norm, lin, lis, lisnot, lsum, lcmp, normfunc


@pytest.mark.unittest
class TestConfigLoaderNorm:

    def test_add(self):
        _norm = norm(item('a')) + 2
        assert _norm({'a': 2}) == 4

        _norm = 3 + norm(item('a'))
        assert _norm({'a': 2}) == 5

        _norm = norm(item('a')) + norm(item('b'))
        assert _norm({'a': 2, 'b': 4}) == 6

    def test_sub(self):
        _norm = norm(item('a')) - 2
        assert _norm({'a': 2}) == 0

        _norm = 3 - norm(item('a'))
        assert _norm({'a': 2}) == 1

        _norm = norm(item('a')) - norm(item('b'))
        assert _norm({'a': 2, 'b': 4}) == -2

    def test_mul(self):
        _norm = norm(item('a')) * 2
        assert _norm({'a': 2}) == 4

        _norm = 3 * norm(item('a'))
        assert _norm({'a': 2}) == 6

        _norm = norm(item('a')) * norm(item('b'))
        assert _norm({'a': 2, 'b': 4}) == 8

    def test_matmul(self):
        # TODO: complete this part
        logging.warning('Testing of matmul for norm not implemented.')

    def test_truediv(self):
        _norm = norm(item('a')) / 2
        assert _norm({'a': 3}) == 1.5

        _norm = 3 / norm(item('a'))
        assert _norm({'a': 2}) == 1.5

        _norm = norm(item('a')) / norm(item('b'))
        assert _norm({'a': 2.1, 'b': 4.2}) == 0.5

    def test_floordiv(self):
        _norm = norm(item('a')) // 2
        assert _norm({'a': 3}) == 1

        _norm = 3 // norm(item('a'))
        assert _norm({'a': 2}) == 1

        _norm = norm(item('a')) // norm(item('b'))
        assert _norm({'a': 10.5, 'b': 4.2}) == 2

    def test_mod(self):
        _norm = norm(item('a')) % 3
        assert _norm({'a': 2}) == 2
        assert _norm({'a': 4}) == 1

        _norm = 4 % norm(item('a'))
        assert _norm({'a': 2}) == 0
        assert _norm({'a': 3}) == 1

        _norm = norm(item('a')) % norm(item('b'))
        assert _norm({'a': 3, 'b': 2}) == 1
        assert _norm({'a': 5, 'b': 3}) == 2

    def test_pow(self):
        _norm = norm(item('a')) ** 3
        assert _norm({'a': 2}) == 8
        assert _norm({'a': 4}) == 64

        _norm = 4 ** norm(item('a'))
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 3}) == 64

        _norm = norm(item('a')) ** norm(item('b'))
        assert _norm({'a': 3, 'b': 2}) == 9
        assert _norm({'a': 5, 'b': 3}) == 125

    def test_lshift(self):
        _norm = norm(item('a')) << 3
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 4}) == 32

        _norm = 4 << norm(item('a'))
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 3}) == 32

        _norm = norm(item('a')) << norm(item('b'))
        assert _norm({'a': 3, 'b': 2}) == 12
        assert _norm({'a': 5, 'b': 3}) == 40

    def test_rshift(self):
        _norm = norm(item('a')) >> 3
        assert _norm({'a': 283}) == 35
        assert _norm({'a': 47}) == 5

        _norm = 47 >> norm(item('a'))
        assert _norm({'a': 2}) == 11
        assert _norm({'a': 3}) == 5

        _norm = norm(item('a')) >> norm(item('b'))
        assert _norm({'a': 37, 'b': 2}) == 9
        assert _norm({'a': 529, 'b': 5}) == 16

    def test_and(self):
        _norm = norm(item('a')) & 9
        assert _norm({'a': 15}) == 9
        assert _norm({'a': 1}) == 1

        _norm = 11 & norm(item('a'))
        assert _norm({'a': 15}) == 11
        assert _norm({'a': 7}) == 3

        _norm = norm(item('a')) & norm(item('b'))
        assert _norm({'a': 15, 'b': 11}) == 11
        assert _norm({'a': 9, 'b': 1}) == 1

    def test_or(self):
        _norm = norm(item('a')) | 9
        assert _norm({'a': 15}) == 15
        assert _norm({'a': 83}) == 91

        _norm = 11 | norm(item('a'))
        assert _norm({'a': 15}) == 15
        assert _norm({'a': 17}) == 27

        _norm = norm(item('a')) | norm(item('b'))
        assert _norm({'a': 5, 'b': 11}) == 15
        assert _norm({'a': 9, 'b': 3}) == 11

    def test_xor(self):
        _norm = norm(item('a')) ^ 9
        assert _norm({'a': 15}) == 6
        assert _norm({'a': 83}) == 90

        _norm = 11 ^ norm(item('a'))
        assert _norm({'a': 15}) == 4
        assert _norm({'a': 17}) == 26

        _norm = norm(item('a')) ^ norm(item('b'))
        assert _norm({'a': 5, 'b': 11}) == 14
        assert _norm({'a': 9, 'b': 3}) == 10

    def test_invert(self):
        _norm = ~norm(item('a'))
        assert _norm({'a': 15}) == -16
        assert _norm({'a': -2348}) == 2347

    def test_pos(self):
        _norm = +norm(item('a'))
        assert _norm({'a': 15}) == 15
        assert _norm({'a': -2348}) == -2348

    def test_neg(self):
        _norm = -norm(item('a'))
        assert _norm({'a': 15}) == -15
        assert _norm({'a': -2348}) == 2348

    def test_eq(self):
        _norm = norm(item('a')) == 2
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 == norm(item('a'))
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(item('a')) == norm(item('b'))
        assert _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 2, 'b': 3})

    def test_ne(self):
        _norm = norm(item('a')) != 2
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 != norm(item('a'))
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(item('a')) != norm(item('b'))
        assert not _norm({'a': 2, 'b': 2})
        assert _norm({'a': 2, 'b': 3})

    def test_lt(self):
        _norm = norm(item('a')) < 2
        assert _norm({'a': 1})
        assert not _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 < norm(item('a'))
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(item('a')) < norm(item('b'))
        assert _norm({'a': 1, 'b': 2})
        assert not _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 3, 'b': 2})

    def test_le(self):
        _norm = norm(item('a')) <= 2
        assert _norm({'a': 1})
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 <= norm(item('a'))
        assert not _norm({'a': 1})
        assert _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(item('a')) <= norm(item('b'))
        assert _norm({'a': 1, 'b': 2})
        assert _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 3, 'b': 2})

    def test_gt(self):
        _norm = norm(item('a')) > 2
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 > norm(item('a'))
        assert _norm({'a': 1})
        assert not _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(item('a')) > norm(item('b'))
        assert not _norm({'a': 1, 'b': 2})
        assert not _norm({'a': 2, 'b': 2})
        assert _norm({'a': 3, 'b': 2})

    def test_ge(self):
        _norm = norm(item('a')) >= 2
        assert not _norm({'a': 1})
        assert _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 >= norm(item('a'))
        assert _norm({'a': 1})
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(item('a')) >= norm(item('b'))
        assert not _norm({'a': 1, 'b': 2})
        assert _norm({'a': 2, 'b': 2})
        assert _norm({'a': 3, 'b': 2})

    def test_lin(self):
        _norm = lin(norm(item('a')), 'string')
        assert _norm({'a': ['string', 1, 2]})
        assert not _norm({'a': ['strng', 1, 2]})

        _norm = lin([1, 2, 3], norm(item('a')))
        assert _norm({'a': 1})
        assert not _norm({'a': 4})

        _norm = lin(norm(item('a')), norm(item('b')))
        assert _norm({'a': [1, 2], 'b': 1})
        assert not _norm({'a': [1, 2], 'b': 3})

    def test_lis(self):
        _norm = lis(norm(item('a')), 'string')
        assert _norm({'a': 'string'})
        assert not _norm({'a': ['strng', 1, 2]})

        _norm = lis(None, norm(item('a')))
        assert _norm({'a': None})
        assert not _norm({'a': 4})

        _norm = lis(norm(item('a')), norm(item('b')))
        assert _norm({'a': 1, 'b': 1})
        assert not _norm({'a': [1, 2], 'b': 3})

    def test_lisnot(self):
        _norm = lisnot(norm(item('a')), 'string')
        assert not _norm({'a': 'string'})
        assert _norm({'a': ['strng', 1, 2]})

        _norm = lisnot(None, norm(item('a')))
        assert not _norm({'a': None})
        assert _norm({'a': 4})

        _norm = lisnot(norm(item('a')), norm(item('b')))
        assert not _norm({'a': 1, 'b': 1})
        assert _norm({'a': [1, 2], 'b': 3})

    def test_lsum(self):
        _norm = lsum(1, 2, norm(item('a') | item('b')), norm(item('c')))
        assert _norm({'a': 1, 'c': 10}) == 14
        assert _norm({'b': 20, 'c': 100}) == 123
        assert _norm({'b': 20, 'a': 30, 'c': -1}) == 32

    def test_lcmp(self):
        _norm = lcmp(2, '<', norm(item('a')), "<=", 5)
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})
        assert _norm({'a': 4})
        assert _norm({'a': 5})
        assert not _norm({'a': 6})

        _norm = lcmp(2, '>=', norm(item('b')), '>', -1)
        assert not _norm({'b': -2})
        assert not _norm({'b': -1})
        assert _norm({'b': 0})
        assert _norm({'b': 1})
        assert _norm({'b': 2})
        assert not _norm({'b': 3})

        _norm = lcmp(2, '!=', norm(item('c')), '==', 1)
        assert _norm({'c': 1})
        assert not _norm({'c': 2})

    def test_lcmp_invalid(self):
        _norm = lcmp(2, '<', norm(item('a')), "<x", 5)
        with pytest.raises(KeyError):
            _norm({'a': 2})

        _norm = lcmp(2, '<', norm(item('a')), "<=")
        with pytest.raises(ValueError):
            _norm({'a': 2})

    def test_norm_complex_case_1(self):
        _norm = norm(item('a')) * (norm(item('b')) - norm(item('c'))) * 2 == norm(item('result') | item('sum'))

        assert not _norm({'a': 2, 'b': 10, 'c': -2, 'result': 1})
        assert _norm({'a': 2, 'b': 10, 'c': -2, 'result': 48})
        with pytest.raises(KeyError):
            _norm({'a': 2, 'b': 10, 'cc': -2, 'result': 24})

        assert not _norm({'a': 2, 'b': 10, 'c': -2, 'sum': 1})
        assert _norm({'a': 2, 'b': 10, 'c': -2, 'result_': 23, 'sum': 48})

    def test_norm_complex_case_2(self):

        def _check(a, b, s):
            return (a + b == s) and (a * b == s)

        _norm = normfunc(_check)(norm(item('a')), norm(item('b')), norm(item('sum')))
        assert not _norm({'a': 1, 'b': 2, 'sum': 3})
        assert _norm({'a': 2, 'b': 2, 'sum': 4})

    def test_norm_back_to_loader(self):
        _loader = Loader(norm(item('a')) + norm(item('b'))) >> interval(1, 3)
        assert _loader({'a': 2, 'b': -1}) == 1
        assert _loader({'a': 1, 'b': 1}) == 2
        with pytest.raises(ValueError):
            _loader({'a': 0, 'b': 0})
        with pytest.raises(ValueError):
            _loader({'a': 0, 'b': 10})
        with pytest.raises(KeyError):
            _loader({'a': 0, 'bb': 2})
