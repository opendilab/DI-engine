from functools import partial
from typing import Union

import pytest

from nervex.utils.autolog import LoggedModel, LoggedValue, TickTime


@pytest.mark.unittest
class TestAutologModel:
    # noinspection DuplicatedCode
    def __get_demo_class(self):

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

    def test_autolog_model(self):
        _class = self.__get_demo_class()

        tick_time = TickTime()
        _tick_monitor = _class(tick_time, expire=10)

        _assert_results = [
            (0.0, 0.0),
            (0.1, 0.2),
            (0.3, 0.6),
            (0.6, 1.2),
            (1.0, 2.0),
            (1.5, 3.0),
        ]

        for i in range(0, 6):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            tick_time.step()

            _thin, _thout = _tick_monitor.thruput['in_time'](), _tick_monitor.thruput['out_time']()
            _exp_thin, _exp_thout = _assert_results[i]

            assert _thin == _exp_thin
            assert _thout == _exp_thout
