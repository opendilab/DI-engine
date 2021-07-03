import time
from unittest.mock import Mock

import pytest

from ding.utils.autolog import TickTime, NaturalTime, TimeProxy


class TestNaturalTime:

    @pytest.mark.unittest
    def test_natural_time(self):
        _time = NaturalTime()
        assert abs(_time.time() - time.time()) < 0.2

    @pytest.mark.benchmark
    def test_natural_time_for_100k_times(self):
        for i in range(0, 100000):
            _time = NaturalTime()
            assert abs(_time.time() - time.time()) < 0.2

    @pytest.mark.unittest
    def test_natural_time_with_mad_system(self):
        _time_func, time.time = time.time, Mock(side_effect=[1.5, 1.8, 2.0, 2.0, 1.75, 1.9, 2.2])

        try:
            _time = NaturalTime()
            assert _time.time() == 1.5
            assert _time.time() == 1.8
            assert _time.time() == 2.0
            assert _time.time() == 2.0
            assert _time.time() == 2.0
            assert _time.time() == 2.0
            assert _time.time() == 2.2
        finally:
            time.time = _time_func


class TestTickTime:

    @pytest.mark.unittest
    def test_tick_bare(self):
        _time = TickTime()
        assert _time.time() == 0
        assert _time.step() == 1
        assert _time.time() == 1
        assert _time.step(2) == 3
        assert _time.time() == 3

        with pytest.raises(TypeError):
            _time.step(0.9)

        with pytest.raises(ValueError):
            _time.step(0)

    @pytest.mark.unittest
    def test_tick_init(self):
        _time = TickTime(3)
        assert _time.time() == 3
        assert _time.step() == 4
        assert _time.time() == 4
        assert _time.step(2) == 6
        assert _time.time() == 6

        with pytest.raises(TypeError):
            _time.step(0.9)

        with pytest.raises(ValueError):
            _time.step(0)


class TestTimeProxy:

    @pytest.mark.unittest
    def test_time_proxy_for_tick_time(self):
        _time = TickTime()
        _proxy = TimeProxy(_time)

        assert _proxy.time() == 0
        assert _proxy.current_time() == 0
        assert not _proxy.is_frozen

        _time.step()
        assert _proxy.time() == 1
        assert _proxy.current_time() == 1
        assert not _proxy.is_frozen

        _proxy.freeze()
        _time.step(2)
        assert _proxy.time() == 1
        assert _proxy.current_time() == 3
        assert _proxy.is_frozen

        _time.step()
        assert _proxy.time() == 1
        assert _proxy.current_time() == 4
        assert _proxy.is_frozen

        _proxy.unfreeze()
        assert _proxy.time() == 4
        assert _proxy.current_time() == 4
        assert not _proxy.is_frozen

    @pytest.mark.unittest
    def test_time_proxy_frozen_for_tick_time(self):
        _time = TickTime()
        _proxy = TimeProxy(_time, frozen=True)

        assert _proxy.time() == 0
        assert _proxy.current_time() == 0
        assert _proxy.is_frozen

        _time.step()
        assert _proxy.time() == 0
        assert _proxy.current_time() == 1
        assert _proxy.is_frozen

        _time.step(2)
        assert _proxy.time() == 0
        assert _proxy.current_time() == 3
        assert _proxy.is_frozen

        _time.step()
        assert _proxy.time() == 0
        assert _proxy.current_time() == 4
        assert _proxy.is_frozen

        _proxy.unfreeze()
        assert _proxy.time() == 4
        assert _proxy.current_time() == 4
        assert not _proxy.is_frozen
