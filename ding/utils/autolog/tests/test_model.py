import time
from functools import partial
from typing import Union

import pytest

from ding.utils.autolog import LoggedModel, LoggedValue, TickTime, NaturalTime, TimeMode


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestAutologModel:

    def __get_demo_class(self):
        # noinspection DuplicatedCode
        class _TickModel(LoggedModel):
            in_time = LoggedValue(float)
            out_time = LoggedValue(float)
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
                    self.register_attribute_value(
                        'reversed_name', _prop_name, partial(lambda name: name[::-1], _prop_name)
                    )

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

    def test_property_getter(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        assert _tick_monitor.reversed_name['in_time']() == 'emit_ni'
        assert _tick_monitor.reversed_name['out_time']() == 'emit_tuo'

        with pytest.raises(KeyError):
            _tick_monitor.reversed_name['property_not_exist']()
        with pytest.raises(KeyError):
            _tick_monitor.reversed_nam['in_time']()

    def test_time(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        assert id(_tick_monitor.time) == id(_time)
        assert _tick_monitor.fixed_time() == 0
        assert _tick_monitor.current_time() == 0

        _tick_monitor.freeze()
        _time.step()
        assert _tick_monitor.fixed_time() == 0
        assert _tick_monitor.current_time() == 1

        _tick_monitor.unfreeze()
        assert _tick_monitor.fixed_time() == 1
        assert _tick_monitor.current_time() == 1

    def test_expire(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        assert _tick_monitor.expire == 5

    def test_with_tick_time(self):
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

    def test_with_natural_time(self):
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

    def test_double_model(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor_1 = _class(_time, expire=5)
        _tick_monitor_2 = _class(_time, expire=5)

        _assert_results_1 = [
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
        _assert_results_2 = [
            (0.0, 0.0), (0.4, 0.8), (1.2, 2.4), (2.4, 4.8), (4.0, 8.0), (6.0, 12.0), (8.4, 16.8), (10.8, 21.6),
            (13.2, 26.4), (15.6, 31.2)
        ]

        for i in range(0, 10):
            _tick_monitor_1.in_time = 1.0 * i
            _tick_monitor_1.out_time = 2.0 * i
            _tick_monitor_2.in_time = 2.0 * i
            _tick_monitor_2.out_time = 4.0 * i

            _time.step()

            _thin_1, _thout_1 = _tick_monitor_1.thruput['in_time'](), _tick_monitor_1.thruput['out_time']()
            _exp_thin_1, _exp_thout_1 = _assert_results_1[i]

            _thin_2, _thout_2 = _tick_monitor_2.thruput['in_time'](), _tick_monitor_2.thruput['out_time']()
            _exp_thin_2, _exp_thout_2 = _assert_results_2[i]

            assert (_thin_1, _thout_1) == (_exp_thin_1, _exp_thout_1)
            assert (_thin_2, _thout_2) == (_exp_thin_2, _exp_thout_2)

    def test_range_values_default(self):
        _class = self.__get_demo_class()

        _time = TickTime()
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            ([((0, 1), 0.0)], [((0, 1), 0.0)]),
            ([((0, 1), 0.0), ((1, 2), 1.0)], [((0, 1), 0.0), ((1, 2), 2.0)]),
            ([((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0)]),
            (
                [((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0),
                 ((3, 4), 3.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0)]
            ),
            (
                [((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0),
                 ((4, 5), 4.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0)]
            ),
            (
                [((1, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0)], [
                    ((1, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0)
                ]
            ),
            (
                [((2, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0)], [
                    ((2, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0)
                ]
            ),
            (
                [((3, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0)], [
                    ((3, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0)
                ]
            ),
            (
                [((4, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0), ((8, 9), 8.0)], [
                    ((4, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0), ((8, 9), 16.0)
                ]
            ),
            (
                [((5, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0), ((8, 9), 8.0), ((9, 10), 9.0)], [
                    ((5, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0), ((8, 9), 16.0), ((9, 10), 18.0)
                ]
            ),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            _time.step()

            _thin, _thout = _tick_monitor.range_values['in_time'](), _tick_monitor.range_values['out_time']()
            _exp_thin, _exp_thout = _assert_results[i]

            assert (_thin, _thout) == (_exp_thin, _exp_thout)

    def test_range_values_absolute(self):
        _class = self.__get_demo_class()

        _time = TickTime(1)
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            ([((1, 2), 0.0)], [((1, 2), 0.0)]),
            ([((1, 2), 0.0), ((2, 3), 1.0)], [((1, 2), 0.0), ((2, 3), 2.0)]),
            ([((1, 2), 0.0), ((2, 3), 1.0), ((3, 4), 2.0)], [((1, 2), 0.0), ((2, 3), 2.0), ((3, 4), 4.0)]),
            (
                [((1, 2), 0.0), ((2, 3), 1.0), ((3, 4), 2.0),
                 ((4, 5), 3.0)], [((1, 2), 0.0), ((2, 3), 2.0), ((3, 4), 4.0), ((4, 5), 6.0)]
            ),
            (
                [((1, 2), 0.0), ((2, 3), 1.0), ((3, 4), 2.0), ((4, 5), 3.0),
                 ((5, 6), 4.0)], [((1, 2), 0.0), ((2, 3), 2.0), ((3, 4), 4.0), ((4, 5), 6.0), ((5, 6), 8.0)]
            ),
            (
                [((2, 2), 0.0), ((2, 3), 1.0), ((3, 4), 2.0), ((4, 5), 3.0), ((5, 6), 4.0), ((6, 7), 5.0)], [
                    ((2, 2), 0.0), ((2, 3), 2.0), ((3, 4), 4.0), ((4, 5), 6.0), ((5, 6), 8.0), ((6, 7), 10.0)
                ]
            ),
            (
                [((3, 3), 1.0), ((3, 4), 2.0), ((4, 5), 3.0), ((5, 6), 4.0), ((6, 7), 5.0), ((7, 8), 6.0)], [
                    ((3, 3), 2.0), ((3, 4), 4.0), ((4, 5), 6.0), ((5, 6), 8.0), ((6, 7), 10.0), ((7, 8), 12.0)
                ]
            ),
            (
                [((4, 4), 2.0), ((4, 5), 3.0), ((5, 6), 4.0), ((6, 7), 5.0), ((7, 8), 6.0), ((8, 9), 7.0)], [
                    ((4, 4), 4.0), ((4, 5), 6.0), ((5, 6), 8.0), ((6, 7), 10.0), ((7, 8), 12.0), ((8, 9), 14.0)
                ]
            ),
            (
                [((5, 5), 3.0), ((5, 6), 4.0), ((6, 7), 5.0), ((7, 8), 6.0), ((8, 9), 7.0), ((9, 10), 8.0)], [
                    ((5, 5), 6.0), ((5, 6), 8.0), ((6, 7), 10.0), ((7, 8), 12.0), ((8, 9), 14.0), ((9, 10), 16.0)
                ]
            ),
            (
                [((6, 6), 4.0), ((6, 7), 5.0), ((7, 8), 6.0), ((8, 9), 7.0), ((9, 10), 8.0), ((10, 11), 9.0)], [
                    ((6, 6), 8.0), ((6, 7), 10.0), ((7, 8), 12.0), ((8, 9), 14.0), ((9, 10), 16.0), ((10, 11), 18.0)
                ]
            ),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            _time.step()

            _thin = _tick_monitor.range_values['in_time'](TimeMode.ABSOLUTE)
            _thout = _tick_monitor.range_values['out_time'](TimeMode.ABSOLUTE)
            _exp_thin, _exp_thout = _assert_results[i]

            assert (_thin, _thout) == (_exp_thin, _exp_thout)

    def test_range_values_lifecycle(self):
        _class = self.__get_demo_class()

        _time = TickTime(1)
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            ([((0, 1), 0.0)], [((0, 1), 0.0)]),
            ([((0, 1), 0.0), ((1, 2), 1.0)], [((0, 1), 0.0), ((1, 2), 2.0)]),
            ([((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0)]),
            (
                [((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0),
                 ((3, 4), 3.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0)]
            ),
            (
                [((0, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0),
                 ((4, 5), 4.0)], [((0, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0)]
            ),
            (
                [((1, 1), 0.0), ((1, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0)], [
                    ((1, 1), 0.0), ((1, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0)
                ]
            ),
            (
                [((2, 2), 1.0), ((2, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0)], [
                    ((2, 2), 2.0), ((2, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0)
                ]
            ),
            (
                [((3, 3), 2.0), ((3, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0)], [
                    ((3, 3), 4.0), ((3, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0)
                ]
            ),
            (
                [((4, 4), 3.0), ((4, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0), ((8, 9), 8.0)], [
                    ((4, 4), 6.0), ((4, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0), ((8, 9), 16.0)
                ]
            ),
            (
                [((5, 5), 4.0), ((5, 6), 5.0), ((6, 7), 6.0), ((7, 8), 7.0), ((8, 9), 8.0), ((9, 10), 9.0)], [
                    ((5, 5), 8.0), ((5, 6), 10.0), ((6, 7), 12.0), ((7, 8), 14.0), ((8, 9), 16.0), ((9, 10), 18.0)
                ]
            ),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            _time.step()

            # print('(', _tick_monitor.range_values['in_time'](TimeMode.RELATIVE_LIFECYCLE), ',',
            #       _tick_monitor.range_values['out_time'](TimeMode.RELATIVE_LIFECYCLE), '),')

            _thin = _tick_monitor.range_values['in_time'](TimeMode.RELATIVE_LIFECYCLE)
            _thout = _tick_monitor.range_values['out_time'](TimeMode.RELATIVE_LIFECYCLE)
            _exp_thin, _exp_thout = _assert_results[i]

            assert (_thin, _thout) == (_exp_thin, _exp_thout)

    def test_range_values_current(self):
        _class = self.__get_demo_class()

        _time = TickTime(1)
        _tick_monitor = _class(_time, expire=5)

        _assert_results = [
            ([((-1, 0), 0.0)], [((-1, 0), 0.0)]),
            ([((-2, -1), 0.0), ((-1, 0), 1.0)], [((-2, -1), 0.0), ((-1, 0), 2.0)]),
            ([((-3, -2), 0.0), ((-2, -1), 1.0), ((-1, 0), 2.0)], [((-3, -2), 0.0), ((-2, -1), 2.0), ((-1, 0), 4.0)]),
            (
                [((-4, -3), 0.0), ((-3, -2), 1.0), ((-2, -1), 2.0),
                 ((-1, 0), 3.0)], [((-4, -3), 0.0), ((-3, -2), 2.0), ((-2, -1), 4.0), ((-1, 0), 6.0)]
            ),
            (
                [((-5, -4), 0.0), ((-4, -3), 1.0), ((-3, -2), 2.0), ((-2, -1), 3.0),
                 ((-1, 0), 4.0)], [((-5, -4), 0.0), ((-4, -3), 2.0), ((-3, -2), 4.0), ((-2, -1), 6.0), ((-1, 0), 8.0)]
            ),
            (
                [((-5, -5), 0.0), ((-5, -4), 1.0), ((-4, -3), 2.0), ((-3, -2), 3.0), ((-2, -1), 4.0), ((-1, 0), 5.0)], [
                    ((-5, -5), 0.0), ((-5, -4), 2.0), ((-4, -3), 4.0), ((-3, -2), 6.0), ((-2, -1), 8.0),
                    ((-1, 0), 10.0)
                ]
            ),
            (
                [((-5, -5), 1.0), ((-5, -4), 2.0), ((-4, -3), 3.0), ((-3, -2), 4.0), ((-2, -1), 5.0), ((-1, 0), 6.0)], [
                    ((-5, -5), 2.0), ((-5, -4), 4.0), ((-4, -3), 6.0), ((-3, -2), 8.0), ((-2, -1), 10.0),
                    ((-1, 0), 12.0)
                ]
            ),
            (
                [((-5, -5), 2.0), ((-5, -4), 3.0), ((-4, -3), 4.0), ((-3, -2), 5.0), ((-2, -1), 6.0), ((-1, 0), 7.0)], [
                    ((-5, -5), 4.0), ((-5, -4), 6.0), ((-4, -3), 8.0), ((-3, -2), 10.0), ((-2, -1), 12.0),
                    ((-1, 0), 14.0)
                ]
            ),
            (
                [((-5, -5), 3.0), ((-5, -4), 4.0), ((-4, -3), 5.0), ((-3, -2), 6.0), ((-2, -1), 7.0), ((-1, 0), 8.0)], [
                    ((-5, -5), 6.0), ((-5, -4), 8.0), ((-4, -3), 10.0), ((-3, -2), 12.0), ((-2, -1), 14.0),
                    ((-1, 0), 16.0)
                ]
            ),
            (
                [((-5, -5), 4.0), ((-5, -4), 5.0), ((-4, -3), 6.0), ((-3, -2), 7.0), ((-2, -1), 8.0), ((-1, 0), 9.0)], [
                    ((-5, -5), 8.0), ((-5, -4), 10.0), ((-4, -3), 12.0), ((-3, -2), 14.0), ((-2, -1), 16.0),
                    ((-1, 0), 18.0)
                ]
            ),
        ]

        for i in range(0, 10):
            _tick_monitor.in_time = 1.0 * i
            _tick_monitor.out_time = 2.0 * i
            _time.step()

            # print('(', _tick_monitor.range_values['in_time'](TimeMode.RELATIVE_CURRENT_TIME), ',',
            #       _tick_monitor.range_values['out_time'](TimeMode.RELATIVE_CURRENT_TIME), '),')

            _thin = _tick_monitor.range_values['in_time'](TimeMode.RELATIVE_CURRENT_TIME)
            _thout = _tick_monitor.range_values['out_time'](TimeMode.RELATIVE_CURRENT_TIME)
            _exp_thin, _exp_thout = _assert_results[i]

            assert (_thin, _thout) == (_exp_thin, _exp_thout)
