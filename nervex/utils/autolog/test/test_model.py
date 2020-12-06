import time
from functools import partial
from typing import Union

import pytest

from nervex.utils.autolog import LoggedModel, LoggedValue, TickTime, NaturalTime


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestAutologModel:
    def __get_demo_class(self):
        # noinspection DuplicatedCode
        class _TickModel(LoggedModel):
            in_time = LoggedValue('in_time', float)
            out_time = LoggedValue('out_time', float)
            __thruput_property_names = ['in_time', 'out_time']

            def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
                LoggedModel.__init__(self, time_, expire)
                self.__register()

            def __register(self):
                def __avg_func(prop_name: str) -> float:
                    records = self.range_values[prop_name]()
                    _sum = sum([_value for (_begin_time, _end_time), _value in records])
                    return _sum / self.expire

                for _prop_name in self.__thruput_property_names:
                    self.register_attribute_value('thruput', _prop_name, partial(__avg_func, _prop_name))

        return _TickModel

    def test_getter_and_setter(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        with pytest.raises(ValueError):
            _ = _tick_monitor.in_time
        with pytest.raises(ValueError):
            _ = _tick_monitor.out_time

        _tick_monitor.in_time = 2.0
        assert _tick_monitor.in_time == 2.0

        with pytest.raises(TypeError):
            _tick_monitor.in_time = None
        assert _tick_monitor.in_time == 2.0

    def test_time(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        assert _tick_monitor.time == _time.time()

    def test_expire(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        assert _tick_monitor.expire == 5

    def test_autolog_model_with_tick_time(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.6, 1.2),
            (1.2, 2.4),
            (2.0, 4.0),
            (3.0, 6.0),
            (4.2, 8.4),
            (5.4, 10.8),
            (6.6, 13.2),
            (7.8, 15.6),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            _time.step()

            _thin, _thout = _tick_monitor.thruput['in_time'](), _tick_monitor.thruput['out_time']()
            _exp_thin, _exp_thout = _assert_results[i]

            assert _thin == _exp_thin
            assert _thout == _exp_thout

    def test_autolog_model_with_natural_time(self):
        _class = self.__get_demo_class()

        _time = NaturalTime()
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.6, 1.2),
            (1.2, 2.4),
            (2.0, 4.0),
            (3.0, 6.0),
            (4.0, 8.0),
            (5.0, 10.0),
            (6.0, 12.0),
            (7.0, 14.0),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            time.sleep(1.0)

            _thin, _thout = _tick_monitor.thruput['in_time'](), _tick_monitor.thruput['out_time']()
            _exp_thin, _exp_thout = _assert_results[i]

            assert abs(_thin - _exp_thin) < 0.1
            assert abs(_thout - _exp_thout) < 0.1
