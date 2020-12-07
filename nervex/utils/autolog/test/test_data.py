import pytest

from nervex.utils.autolog import TimeRangedData, NaturalTime, TickTime


@pytest.mark.unittest
class TestAutologRangedData:
    def test_expire(self):
        data = TimeRangedData(NaturalTime(), expire=5)
        assert data.expire == 5

        with pytest.raises(ValueError):
            TimeRangedData(NaturalTime(), expire=-1)

        with pytest.raises(TypeError):
            TimeRangedData(NaturalTime(), expire='5')

    def test_len(self):
        data = TimeRangedData(TickTime(), expire=5)
        assert len(data) == 0

        data.append(233)
        assert len(data) == 1

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert len(data) == 5

        data.time.step(4)
        assert len(data) == 5

        data.time.step(1)
        assert len(data) == 4

        data.time.step(10)
        assert len(data) == 0

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

        data.time.step(10)
        assert not data

    def test_iter(self):
        data = TimeRangedData(TickTime(), expire=5)
        assert list(data) == []

        data.append(233)
        assert list(data) == [233]

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert list(data) == [233, 2, 3, 5, 7]

        data.time.step(4)
        assert list(data) == [233, 2, 3, 5, 7]

        data.time.step(1)
        assert list(data) == [2, 3, 5, 7]

        data.time.step(10)
        assert list(data) == []

    def test_getitem(self):
        data = TimeRangedData(TickTime(), expire=5)
        data.append(233)
        assert data[0] == 233
        assert data[-1] == 233

        data.time.step()
        data.extend([2, 3, 5, 7])
        assert data[0] == 233
        assert data[-1] == 7
        assert data[1] == 2

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

        data.time.step(10)
        with pytest.raises(ValueError):
            _ = data.current()
