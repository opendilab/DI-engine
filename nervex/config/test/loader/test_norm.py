import warnings

import pytest

from ...loader.mapping import index
from ...loader.norm import norm, lin, lis, lisnot, lsum, lcmp


@pytest.mark.unittest
class TestConfigLoaderNorm:

    def test_norm_general(self):
        _norm = norm(index('a')) * (norm(index('b')) - norm(index('c'))) * 2 == norm(index('result') | index('sum'))

        assert not _norm({'a': 2, 'b': 10, 'c': -2, 'result': 1})
        assert _norm({'a': 2, 'b': 10, 'c': -2, 'result': 48})
        with pytest.raises(KeyError):
            _norm({'a': 2, 'b': 10, 'cc': -2, 'result': 24})

        assert not _norm({'a': 2, 'b': 10, 'c': -2, 'sum': 1})
        assert _norm({'a': 2, 'b': 10, 'c': -2, 'result_': 23, 'sum': 48})

    def test_add(self):
        _norm = norm(index('a')) + 2
        assert _norm({'a': 2}) == 4

        _norm = 3 + norm(index('a'))
        assert _norm({'a': 2}) == 5

        _norm = norm(index('a')) + norm(index('b'))
        assert _norm({'a': 2, 'b': 4}) == 6

    def test_sub(self):
        _norm = norm(index('a')) - 2
        assert _norm({'a': 2}) == 0

        _norm = 3 - norm(index('a'))
        assert _norm({'a': 2}) == 1

        _norm = norm(index('a')) - norm(index('b'))
        assert _norm({'a': 2, 'b': 4}) == -2

    def test_mul(self):
        _norm = norm(index('a')) * 2
        assert _norm({'a': 2}) == 4

        _norm = 3 * norm(index('a'))
        assert _norm({'a': 2}) == 6

        _norm = norm(index('a')) * norm(index('b'))
        assert _norm({'a': 2, 'b': 4}) == 8

    def test_matmul(self):
        # TODO: complete this part
        warnings.warn('Testing of matmul for norm not implemented.')

    def test_truediv(self):
        _norm = norm(index('a')) / 2
        assert _norm({'a': 3}) == 1.5

        _norm = 3 / norm(index('a'))
        assert _norm({'a': 2}) == 1.5

        _norm = norm(index('a')) / norm(index('b'))
        assert _norm({'a': 2.1, 'b': 4.2}) == 0.5

    def test_floordiv(self):
        _norm = norm(index('a')) // 2
        assert _norm({'a': 3}) == 1

        _norm = 3 // norm(index('a'))
        assert _norm({'a': 2}) == 1

        _norm = norm(index('a')) // norm(index('b'))
        assert _norm({'a': 10.5, 'b': 4.2}) == 2

    def test_mod(self):
        _norm = norm(index('a')) % 3
        assert _norm({'a': 2}) == 2
        assert _norm({'a': 4}) == 1

        _norm = 4 % norm(index('a'))
        assert _norm({'a': 2}) == 0
        assert _norm({'a': 3}) == 1

        _norm = norm(index('a')) % norm(index('b'))
        assert _norm({'a': 3, 'b': 2}) == 1
        assert _norm({'a': 5, 'b': 3}) == 2

    def test_pow(self):
        _norm = norm(index('a')) ** 3
        assert _norm({'a': 2}) == 8
        assert _norm({'a': 4}) == 64

        _norm = 4 ** norm(index('a'))
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 3}) == 64

        _norm = norm(index('a')) ** norm(index('b'))
        assert _norm({'a': 3, 'b': 2}) == 9
        assert _norm({'a': 5, 'b': 3}) == 125

    def test_lshift(self):
        _norm = norm(index('a')) << 3
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 4}) == 32

        _norm = 4 << norm(index('a'))
        assert _norm({'a': 2}) == 16
        assert _norm({'a': 3}) == 32

        _norm = norm(index('a')) << norm(index('b'))
        assert _norm({'a': 3, 'b': 2}) == 12
        assert _norm({'a': 5, 'b': 3}) == 40

    def test_rshift(self):
        _norm = norm(index('a')) >> 3
        assert _norm({'a': 283}) == 35
        assert _norm({'a': 47}) == 5

        _norm = 47 >> norm(index('a'))
        assert _norm({'a': 2}) == 11
        assert _norm({'a': 3}) == 5

        _norm = norm(index('a')) >> norm(index('b'))
        assert _norm({'a': 37, 'b': 2}) == 9
        assert _norm({'a': 529, 'b': 5}) == 16

    def test_and(self):
        _norm = norm(index('a')) & 9
        assert _norm({'a': 15}) == 9
        assert _norm({'a': 1}) == 1

        _norm = 11 & norm(index('a'))
        assert _norm({'a': 15}) == 11
        assert _norm({'a': 7}) == 3

        _norm = norm(index('a')) & norm(index('b'))
        assert _norm({'a': 15, 'b': 11}) == 11
        assert _norm({'a': 9, 'b': 1}) == 1

    def test_or(self):
        _norm = norm(index('a')) | 9
        assert _norm({'a': 15}) == 15
        assert _norm({'a': 83}) == 91

        _norm = 11 | norm(index('a'))
        assert _norm({'a': 15}) == 15
        assert _norm({'a': 17}) == 27

        _norm = norm(index('a')) | norm(index('b'))
        assert _norm({'a': 5, 'b': 11}) == 15
        assert _norm({'a': 9, 'b': 3}) == 11

    def test_xor(self):
        _norm = norm(index('a')) ^ 9
        assert _norm({'a': 15}) == 6
        assert _norm({'a': 83}) == 90

        _norm = 11 ^ norm(index('a'))
        assert _norm({'a': 15}) == 4
        assert _norm({'a': 17}) == 26

        _norm = norm(index('a')) ^ norm(index('b'))
        assert _norm({'a': 5, 'b': 11}) == 14
        assert _norm({'a': 9, 'b': 3}) == 10

    def test_invert(self):
        _norm = ~norm(index('a'))
        assert _norm({'a': 15}) == -16
        assert _norm({'a': -2348}) == 2347

    def test_pos(self):
        _norm = +norm(index('a'))
        assert _norm({'a': 15}) == 15
        assert _norm({'a': -2348}) == -2348

    def test_neg(self):
        _norm = -norm(index('a'))
        assert _norm({'a': 15}) == -15
        assert _norm({'a': -2348}) == 2348

    def test_eq(self):
        _norm = norm(index('a')) == 2
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 == norm(index('a'))
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(index('a')) == norm(index('b'))
        assert _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 2, 'b': 3})

    def test_ne(self):
        _norm = norm(index('a')) != 2
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 != norm(index('a'))
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(index('a')) != norm(index('b'))
        assert not _norm({'a': 2, 'b': 2})
        assert _norm({'a': 2, 'b': 3})

    def test_lt(self):
        _norm = norm(index('a')) < 2
        assert _norm({'a': 1})
        assert not _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 < norm(index('a'))
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(index('a')) < norm(index('b'))
        assert _norm({'a': 1, 'b': 2})
        assert not _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 3, 'b': 2})

    def test_le(self):
        _norm = norm(index('a')) <= 2
        assert _norm({'a': 1})
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = 2 <= norm(index('a'))
        assert not _norm({'a': 1})
        assert _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = norm(index('a')) <= norm(index('b'))
        assert _norm({'a': 1, 'b': 2})
        assert _norm({'a': 2, 'b': 2})
        assert not _norm({'a': 3, 'b': 2})

    def test_gt(self):
        _norm = norm(index('a')) > 2
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 > norm(index('a'))
        assert _norm({'a': 1})
        assert not _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(index('a')) > norm(index('b'))
        assert not _norm({'a': 1, 'b': 2})
        assert not _norm({'a': 2, 'b': 2})
        assert _norm({'a': 3, 'b': 2})

    def test_ge(self):
        _norm = norm(index('a')) >= 2
        assert not _norm({'a': 1})
        assert _norm({'a': 2})
        assert _norm({'a': 3})

        _norm = 2 >= norm(index('a'))
        assert _norm({'a': 1})
        assert _norm({'a': 2})
        assert not _norm({'a': 3})

        _norm = norm(index('a')) >= norm(index('b'))
        assert not _norm({'a': 1, 'b': 2})
        assert _norm({'a': 2, 'b': 2})
        assert _norm({'a': 3, 'b': 2})

    def test_lin(self):
        _norm = lin(norm(index('a')), 'string')
        assert _norm({'a': ['string', 1, 2]})
        assert not _norm({'a': ['strng', 1, 2]})

        _norm = lin([1, 2, 3], norm(index('a')))
        assert _norm({'a': 1})
        assert not _norm({'a': 4})

        _norm = lin(norm(index('a')), norm(index('b')))
        assert _norm({'a': [1, 2], 'b': 1})
        assert not _norm({'a': [1, 2], 'b': 3})

    def test_lis(self):
        _norm = lis(norm(index('a')), 'string')
        assert _norm({'a': 'string'})
        assert not _norm({'a': ['strng', 1, 2]})

        _norm = lis(None, norm(index('a')))
        assert _norm({'a': None})
        assert not _norm({'a': 4})

        _norm = lis(norm(index('a')), norm(index('b')))
        assert _norm({'a': 1, 'b': 1})
        assert not _norm({'a': [1, 2], 'b': 3})

    def test_lisnot(self):
        _norm = lisnot(norm(index('a')), 'string')
        assert not _norm({'a': 'string'})
        assert _norm({'a': ['strng', 1, 2]})

        _norm = lisnot(None, norm(index('a')))
        assert not _norm({'a': None})
        assert _norm({'a': 4})

        _norm = lisnot(norm(index('a')), norm(index('b')))
        assert not _norm({'a': 1, 'b': 1})
        assert _norm({'a': [1, 2], 'b': 3})

    def test_lsum(self):
        _norm = lsum(1, 2, norm(index('a') | index('b')), norm(index('c')))
        assert _norm({'a': 1, 'c': 10}) == 14
        assert _norm({'b': 20, 'c': 100}) == 123
        assert _norm({'b': 20, 'a': 30, 'c': -1}) == 32

    def test_lcmp(self):
        _norm = lcmp(2, '<', norm(index('a')), "<=", 5)
        assert not _norm({'a': 1})
        assert not _norm({'a': 2})
        assert _norm({'a': 3})
        assert _norm({'a': 4})
        assert _norm({'a': 5})
        assert not _norm({'a': 6})

    def test_lcmp_invalid(self):
        _norm = lcmp(2, '<', norm(index('a')), "<x", 5)
        with pytest.raises(KeyError):
            _norm({'a': 2})

        _norm = lcmp(2, '<', norm(index('a')), "<=")
        with pytest.raises(ValueError):
            _norm({'a': 2})
