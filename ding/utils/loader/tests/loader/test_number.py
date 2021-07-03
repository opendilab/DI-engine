import math

import pytest

from ding.utils.loader import item, item_or, numeric, interval, negative, plus, minus, minus_with, multi, divide, \
    divide_with, power, power_with, positive, msum, mmulti, mcmp, is_positive, is_negative, non_positive, \
    non_negative, keep


@pytest.mark.unittest
class TestConfigLoaderNumber:
    # noinspection DuplicatedCode
    def test_numeric_plain(self):
        _loader = numeric()

        assert _loader(1) == 1
        assert _loader(1.0) == 1.0
        with pytest.raises(TypeError):
            _loader('1')
        with pytest.raises(TypeError):
            _loader('-1.0')
        assert _loader(math.inf) == math.inf
        assert _loader(-float('inf')) == -math.inf
        with pytest.raises(TypeError):
            _loader('inf')
        with pytest.raises(TypeError):
            _loader('-inf')
        with pytest.raises(ValueError):
            _loader(math.nan)
        with pytest.raises(TypeError):
            _loader('nan')
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader('styring')
        with pytest.raises(TypeError):
            _loader('-abcdef12345')
        with pytest.raises(TypeError):
            _loader('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_int_ban(self):
        _loader = numeric(int_ok=False)
        with pytest.raises(TypeError):
            _loader(1)
        assert _loader(1.0) == 1.0
        with pytest.raises(TypeError):
            _loader('1')
        with pytest.raises(TypeError):
            _loader('-1.0')
        assert _loader(math.inf) == math.inf
        assert _loader(-float('inf')) == -math.inf
        with pytest.raises(TypeError):
            _loader('inf')
        with pytest.raises(TypeError):
            _loader('-inf')
        with pytest.raises(ValueError):
            _loader(math.nan)
        with pytest.raises(TypeError):
            _loader('nan')
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader('styring')
        with pytest.raises(TypeError):
            _loader('-abcdef12345')
        with pytest.raises(TypeError):
            _loader('i n  f')

    # noinspection DuplicatedCode
    def test_numeric_float_ban(self):
        _loader = numeric(float_ok=False)
        assert _loader(1) == 1
        with pytest.raises(TypeError):
            _loader(1.0)
        with pytest.raises(TypeError):
            _loader('1')
        with pytest.raises(TypeError):
            _loader('-1.0')
        with pytest.raises(TypeError):
            _loader(math.inf)
        with pytest.raises(TypeError):
            _loader(-float('inf'))
        with pytest.raises(TypeError):
            _loader('inf')
        with pytest.raises(TypeError):
            _loader('-inf')
        with pytest.raises(ValueError):
            _loader(math.nan)
        with pytest.raises(TypeError):
            _loader('nan')
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader('styring')
        with pytest.raises(TypeError):
            _loader('-abcdef12345')
        with pytest.raises(TypeError):
            _loader('i n  f')

    def test_numeric_double_ban(self):
        with pytest.raises(ValueError):
            numeric(int_ok=False, float_ok=False)

    # noinspection DuplicatedCode
    def test_numeric_inf_ban(self):
        _loader = numeric(inf_ok=False)
        assert _loader(1) == 1
        assert _loader(1.0) == 1.0
        with pytest.raises(TypeError):
            _loader('1')
        with pytest.raises(TypeError):
            _loader('-1.0')
        with pytest.raises(ValueError):
            _loader(math.inf)
        with pytest.raises(ValueError):
            _loader(-float('inf'))
        with pytest.raises(TypeError):
            _loader('inf')
        with pytest.raises(TypeError):
            _loader('-inf')
        with pytest.raises(ValueError):
            _loader(math.nan)
        with pytest.raises(TypeError):
            _loader('nan')
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader('styring')
        with pytest.raises(TypeError):
            _loader('-abcdef12345')
        with pytest.raises(TypeError):
            _loader('i n  f')

    def test_interval_common(self):
        _loader = interval(1, 3.5)
        with pytest.raises(ValueError):
            _loader(0.5)
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        with pytest.raises(ValueError):
            _loader(4.0)

    def test_interval_all(self):
        _loader = interval()
        assert _loader(0.5) == 0.5
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        assert _loader(4.0) == 4.0

    # noinspection DuplicatedCode
    def test_interval_left_open(self):
        _loader = interval(1.0, left_ok=False)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        assert _loader(4.0) == 4.0

    # noinspection DuplicatedCode
    def test_interval_left_open_eps(self):
        _loader = interval(1.0, left_ok=False, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        with pytest.raises(ValueError):
            _loader(1.001)
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        assert _loader(4.0) == 4.0

    # noinspection DuplicatedCode
    def test_interval_left_close(self):
        _loader = interval(1.0)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        assert _loader(4.0) == 4.0

    # noinspection DuplicatedCode
    def test_interval_left_close_eps(self):
        _loader = interval(1.0, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        assert _loader(0.999) == 0.999
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.5) == 3.5
        assert _loader(4.0) == 4.0

    def test_interval_right_open(self):
        _loader = interval(right=3.5, right_ok=False)
        assert _loader(0.5) == 0.5
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_right_open_eps(self):
        _loader = interval(right=3.5, right_ok=False, eps=0.01)
        assert _loader(0.5) == 0.5
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        with pytest.raises(ValueError):
            _loader(3.499)
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    def test_interval_right_close(self):
        _loader = interval(right=3.5)
        assert _loader(0.5) == 0.5
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    def test_interval_right_close_eps(self):
        _loader = interval(right=3.5, eps=0.01)
        assert _loader(0.5) == 0.5
        assert _loader(1.0) == 1.0
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        assert _loader(3.501) == 3.501
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_open_open(self):
        _loader = interval(1.0, 3.5, left_ok=False, right_ok=False)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_open_open_eps(self):
        _loader = interval(1.0, 3.5, left_ok=False, right_ok=False, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        with pytest.raises(ValueError):
            _loader(1.001)
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        with pytest.raises(ValueError):
            _loader(3.499)
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_open_close(self):
        _loader = interval(1.0, 3.5, left_ok=False)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_open_close_eps(self):
        _loader = interval(1.0, 3.5, left_ok=False, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        with pytest.raises(ValueError):
            _loader(1.0)
        with pytest.raises(ValueError):
            _loader(1.001)
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        assert _loader(3.501) == 3.501
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    def test_interval_both_close_open(self):
        _loader = interval(1.0, 3.5, right_ok=False)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_close_open_eps(self):
        _loader = interval(1.0, 3.5, right_ok=False, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        assert _loader(0.999) == 0.999
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        with pytest.raises(ValueError):
            _loader(3.499)
        with pytest.raises(ValueError):
            _loader(3.5)
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_close_close(self):
        _loader = interval(1.0, 3.5)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(0.999)
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        with pytest.raises(ValueError):
            _loader(3.501)
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    # noinspection DuplicatedCode
    def test_interval_both_close_close_eps(self):
        _loader = interval(1.0, 3.5, eps=0.01)
        with pytest.raises(ValueError):
            _loader(0.5)
        with pytest.raises(ValueError):
            _loader(0.9)
        assert _loader(0.999) == 0.999
        assert _loader(1.0) == 1.0
        assert _loader(1.001) == 1.001
        assert _loader(1.1) == 1.1
        assert _loader(1.5) == 1.5
        assert _loader(3.4) == 3.4
        assert _loader(3.499) == 3.499
        assert _loader(3.5) == 3.5
        assert _loader(3.501) == 3.501
        with pytest.raises(ValueError):
            _loader(3.6)
        with pytest.raises(ValueError):
            _loader(4.0)

    def test_interval_invalid(self):
        with pytest.raises(ValueError):
            interval(1.0, 0.9)

    def test_interval_complex_1(self):
        _loader = float & (interval(1, 4, left_ok=False, right_ok=False) | interval(10.2, 13.4, eps=0.01))
        with pytest.raises(ValueError):
            _loader(0.9)
        with pytest.raises(ValueError):
            _loader(1.0)
        assert _loader(1.1) == 1.1
        with pytest.raises(TypeError):
            _loader(2)
        assert _loader(2.0) == 2.0
        assert _loader(3.9) == 3.9
        with pytest.raises(ValueError):
            _loader(4.0)
        with pytest.raises(ValueError):
            _loader(4.1)
        with pytest.raises(ValueError):
            _loader(10.1)
        assert _loader(10.199) == 10.199
        assert _loader(10.2) == 10.2
        with pytest.raises(TypeError):
            _loader(11)
        assert _loader(11.0) == 11.0
        assert _loader(13.4) == 13.4
        assert _loader(13.401) == 13.401
        with pytest.raises(ValueError):
            _loader(13.5)
        with pytest.raises(TypeError):
            _loader(None)
        with pytest.raises(TypeError):
            _loader('string')

    def test_negative(self):
        _loader = negative()
        assert _loader(1) == -1
        assert _loader(-2) == 2

    def test_positive(self):
        _loader = positive()
        assert _loader(1) == 1
        assert _loader(0) == 0
        assert _loader(-1) == -1

    def test_plus(self):
        _loader = plus(1)
        assert _loader(1) == 2
        assert _loader(-2) == -1

        _loader = plus(negative())
        assert _loader(1) == 0
        assert _loader(-2) == 0

    def test_minus(self):
        _loader = minus(2)
        assert _loader(1) == -1
        assert _loader(-2) == -4

        _loader = minus(negative())
        assert _loader(1) == 2
        assert _loader(-2) == -4

    def test_minus_with(self):
        _loader = minus_with(2)
        assert _loader(1) == 1
        assert _loader(-2) == 4

        _loader = minus_with(negative())
        assert _loader(1) == -2
        assert _loader(-2) == 4

    def test_multi(self):
        _loader = multi(2)
        assert _loader(1) == 2
        assert _loader(-2) == -4

        _loader = multi(keep())
        assert _loader(1) == 1
        assert _loader(-2) == 4
        assert _loader(-3) == 9

    def test_divide(self):
        _loader = divide(2)
        assert _loader(1) == 0.5
        assert _loader(-2) == -1

        _loader = divide(negative())
        assert _loader(1) == -1
        assert _loader(-2) == -1

    def test_divide_with(self):
        _loader = divide_with(2)
        assert _loader(1) == 2
        assert _loader(-2) == -1

        _loader = divide_with(negative())
        assert _loader(1) == -1
        assert _loader(-2) == -1

    def test_power(self):
        _loader = power(2)
        assert _loader(1) == 1
        assert _loader(-2) == 4

        _loader = power(keep()) >> power(keep())
        assert _loader(2) == 256
        assert _loader(3) == 443426488243037769948249630619149892803

    def test_power_with(self):
        _loader = power_with(2)
        assert _loader(1) == 2
        assert _loader(-2) == 0.25

        _loader = power_with(minus(1)) >> power_with(minus(1))
        assert _loader(3) == 5764801
        assert _loader(4) == int(
            '14134776518227074636666380005943348126619871175004951664972849610340958208'
            '000000000000000000000000000000000000000000000000000000000000000000000000000000000'
        )

    def test_msum(self):
        _loader = msum(item('a'), item('b'), item_or('c', 0))
        assert _loader({'a': 1, 'b': 3}) == 4
        assert _loader({'a': -2, 'b': 5, 'c': 20}) == 23

    def test_mmulti(self):
        _loader = mmulti(item('a'), item('b'), item_or('c', 1))
        assert _loader({'a': 1, 'b': 3}) == 3
        assert _loader({'a': -2, 'b': 5, 'c': 3}) == -30

    def test_mcmp(self):
        _loader = mcmp(1)
        assert _loader(1) == 1

        _loader = mcmp(1, '<', item('a'), '>=', item('b'))
        assert _loader({'a': 2, 'b': 1}) == {'a': 2, 'b': 1}
        assert _loader({'a': 2, 'b': 2}) == {'a': 2, 'b': 2}
        with pytest.raises(ValueError):
            _loader({'a': 2, 'b': 3})
        with pytest.raises(ValueError):
            _loader({'a': 1, 'b': 0})

        _loader = mcmp(1, '==', keep())
        assert _loader(1) == 1
        with pytest.raises(ValueError):
            _loader(2)

        _loader = mcmp(1, '!=', keep())
        assert _loader(2) == 2
        with pytest.raises(ValueError):
            _loader(1)

        _loader = mcmp(1, '>', item('a'), '<=', item('b'))
        assert _loader({'a': 0, 'b': 1}) == {'a': 0, 'b': 1}
        assert _loader({'a': 0, 'b': 0}) == {'a': 0, 'b': 0}
        with pytest.raises(ValueError):
            _loader({'a': 0, 'b': -1})
        with pytest.raises(ValueError):
            _loader({'a': 1, 'b': 2})

    def test_mcmp_invalid(self):
        with pytest.raises(ValueError):
            mcmp(1, '>', item('a'), '<=', item('b'), '==')
        with pytest.raises(KeyError):
            mcmp(1, '>', item('a'), '*=', item('b'))

    def test_is_positive(self):
        _loader = is_positive()
        assert _loader(1) == 1
        with pytest.raises(ValueError):
            _loader(0)
        with pytest.raises(ValueError):
            _loader(-1)

    def test_is_negative(self):
        _loader = is_negative()
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(0)
        assert _loader(-1) == -1

    def test_non_positive(self):
        _loader = non_positive()
        with pytest.raises(ValueError):
            _loader(1)
        assert _loader(0) == 0
        assert _loader(-1) == -1

    def test_non_negative(self):
        _loader = non_negative()
        assert _loader(1) == 1
        assert _loader(0) == 0
        with pytest.raises(ValueError):
            _loader(-1)
