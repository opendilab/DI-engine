import time
import torch


def build_time_helper(cfg=None, wrapper_type=None):
    # Note: wrapper_type has higher priority
    if cfg is not None:
        time_wrapper_type = cfg.common.time_wrapper_type
    if wrapper_type is not None:
        time_wrapper_type = wrapper_type
    if time_wrapper_type == 'time':
        return TimeWrapperTime
    elif time_wrapper_type == 'cuda':
        return TimeWrapperCuda
    else:
        raise KeyError('invalid time_wrapper_type: {}'.format(time_wrapper_type))


class TimeWrapper(object):

    @classmethod
    def wrapper(cls, fn):
        def time_func(*args, **kwargs):
            cls.start_time()
            ret = fn(*args, **kwargs)
            t = cls.end_time()
            return ret, t
        return time_func

    @classmethod
    def start_time(cls):
        raise NotImplementedError

    @classmethod
    def end_time(cls):
        raise NotImplementedError



class TimeWrapperTime(TimeWrapper):

    # overwrite
    @classmethod
    def start_time(cls):
        cls.start = time.time()

    # overwrite
    @classmethod
    def end_time(cls):
        cls.end = time.time()
        return cls.end - cls.start


class TimeWrapperCuda(TimeWrapper):
    # cls variable is initialized on loading this class
    start_record = torch.cuda.Event(enable_timing=True)
    end_record = torch.cuda.Event(enable_timing=True)

    # overwrite
    @classmethod
    def start_time(cls):
        cls.start = cls.start_record.record()

    # overwrite
    @classmethod
    def end_time(cls):
        cls.end = cls.end_record.record()
        torch.cuda.synchronize()
        return cls.start_record.elapsed_time(cls.end_record) / 1000


def test_time_wrapper():
    class NaiveObject(object):
        pass

    cfg = NaiveObject()
    setattr(cfg, 'common', NaiveObject())
    setattr(cfg.common, 'time_wrapper_type', 'time')
    time_handle = build_time_helper(cfg)

    @time_handle.wrapper
    def func1(x):
        return x + 1

    def func2(x):
        return x + 1

    # usage 1
    ret, t = func1(3)
    print('runtime1', t)

    # usage 2
    time_handle.start_time()
    ret = func2(3)
    t = time_handle.end_time()
    print('runtime2', t)


if __name__ == "__main__":
    test_time_wrapper()
