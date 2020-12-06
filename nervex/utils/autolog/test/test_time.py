import time

import pytest

from nervex.utils.autolog import TickTime, NaturalTime, TimeProxy


@pytest.mark.unittest
class TestNaturalTime:
    def test_natural_time(self):
        for i in range(0, 1000):
            _time = NaturalTime()
            assert abs(_time.time() - time.time()) < 0.2


@pytest.mark.unittest
class TestTickTime:
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


@pytest.mark.unittest
class TestTimeProxy:
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
