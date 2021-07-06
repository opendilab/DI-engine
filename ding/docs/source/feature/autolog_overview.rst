Autolog Overview
========================

概述：
    Autolog模块的主要用途为解决在模型训练等数据动态场景下，数据在有效时间窗口内的自动记录与分析。

    在我们的Autolog中，将时间窗口内的数据及其变化情况进行了完整的记录，并且可供访问，也可供更进一步的封装，为一种方便易用且易配置的日志架构，主打通用性、扩展性和易用性。

    Autolog整体分为两部分：

        - Time：用于将两种时间系统进行封装，供LoggedModel使用。支持以继承等形式的扩展。
        - LoggedModel：用于基于时间系统，以及相关数据字段，进行各个字段的记录和分析。支持以继承等形式的扩展。


Time
-------------

概述：
    时间系统是Autolog系统的关键要素。其用来支撑一个日志模型的记录方式。

    具体来说，Time分为以下两类，均继承自BaseTime类：

        - NaturalTime：自然时间。即系统时间，实型，单位为秒。
        - TickTime：步进时间。即以次数为单位的时间，整型，单位为次。

    BaseTime均提供time()方法以获取时间对象内的时间。

    此外，如有需要，可以自己定义Time类，只需要继承BaseTime并实现time()方法即可用于LoggedModel。


LoggedModel
-------------

概述：
    日志模型系统是Autolog的核心，也是关键所在。其基本用法为继承LoggedModel类，在内部定义LoggedValue的descriptor，并注册一些统计方法，来实现方便易用的统计功能。

    一个简单的demo：

    .. code:: python

        from ding.utils.autolog import LoggedValue, LoggedModel
        class AvgList(LoggedModel):
            value = LoggedValue(float)
            __property_names = ['value']

            def __init__(self, time_: BaseTime, expire: Union[int, float]):
                LoggedModel.__init__(self, time_, expire)

                # attention, original value must be set in __init__ function, or it will not
                # be activated, the timeline of this value will also be unexpectedly affected.
                self.value = 0.0
                self.__register()

            def __register(self):
                def __avg_func(prop_name: str) -> float:  # function to calculate average value of properties
                    records = self.range_values[prop_name]()
                    (_start_time, _), _ = records[0]
                    (_, _end_time), _ = records[-1]

                    _duration = _end_time - _start_time
                    _sum = sum([_value * (_end_time - _begin_time) for (_begin_time, _end_time), _value in records])

                    return _sum / _duration

                for _prop_name in self.__property_names:
                    self.register_attribute_value('avg', _prop_name, partial(__avg_func, prop_name=_prop_name))

    在这样的一个demo中，AvgList类会自动记录对value值的更改，并且结合传入的时间对象进行记录，其内置过期机制，仅保留expire时间内的数据。在__register()函数中，注册了一个用于进行平均值计算的函数。

    以下是上述日志模型的调用例子

    .. code:: python

        from ding.utils.autolog import NaturalTime, TimeMode

        if __name__ == "__main__":
            _time = NaturalTime()
            l = AvgList(_time, expire=10)

            # just do something here ...
            # Such as
            #
            # l.value = 11
            # time.sleep(1.0)  # please add "import time" if you want to use this line
            # l.value = 22
            # time.sleep(1.2)
            # l.value = 33
            #
            # ...
            #
            # These values (such as 11, 22 and 33) will be automatically recorded in l with the natural time (in form of timestamps).

            print(l.range_values['value']()) # original range_values function in LoggedModel of last 10 secs
            print(l.range_values['value'](TimeMode.ABSOLUTE))  # use absolute time
            print(l.avg['value']())  # average value of last 10 secs
