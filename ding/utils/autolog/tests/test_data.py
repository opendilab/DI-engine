import pytest

from ding.utils.autolog import TimeRangedData, NaturalTime, TickTime


@pytest.mark.unittest
class TestAutologRangedData:

    def test_expire(self):
        data = TimeRangedData(NaturalTime(), expire=5)
        assert data.expire == 5

        with pytest.raises(ValueError):
            TimeRangedData(NaturalTime(), expire=-1)

        with pytest.raises(TypeError):
            TimeRangedData(NaturalTime(), expire='5')

    def test_bool(self):
        data = TimeRangedData(TickTime(), expire=5)
        assert not data

        data.append(233)
        assert data

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert data

        data.time.step(4)
        assert data

        data.time.step(1)
        assert data

        data.time.step(1)
        assert data

        data.time.step(1)
        assert data

        data.time.step(10)
        assert data

    def test_current(self):
        data = TimeRangedData(TickTime(), expire=5)
        with pytest.raises(ValueError):
            _ = data.current()

        data.append(233)
        assert data.current() == 233

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert data.current() == 7

        data.time.step(4)
        assert data.current() == 7

        data.time.step(1)
        assert data.current() == 7

        data.time.step(1)
        assert data.current() == 7

        data.time.step(1)
        assert data.current() == 7

        data.time.step(10)
        assert data.current() == 7

    def test_history(self):
        data = TimeRangedData(TickTime(), expire=5)
        assert data.history() == []

        data.append(233)
        assert data.history() == [(0, 233)]

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert data.history() == [(0, 233), (1, 2), (1, 3), (1, 5), (1, 7)]

        data.time.step(4)
        assert data.history() == [(0, 233), (1, 2), (1, 3), (1, 5), (1, 7), (5, 7)]

        data.time.step(1)
        assert data.history() == [(1, 233), (1, 2), (1, 3), (1, 5), (1, 7), (6, 7)]

        data.time.step(1)
        assert data.history() == [(2, 7), (7, 7)]

        data.time.step(1)
        assert data.history() == [(3, 7), (8, 7)]

        data.time.step(10)
        assert data.history() == [(13, 7), (18, 7)]
