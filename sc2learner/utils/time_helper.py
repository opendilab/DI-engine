import time

import torch


def build_time_helper(cfg=None, wrapper_type=None):
    # Note: wrapper_type has higher priority
    if wrapper_type is not None:
        time_wrapper_type = wrapper_type
    elif cfg is not None:
        time_wrapper_type = cfg.common.time_wrapper_type
    if time_wrapper_type == 'time' or (not torch.cuda.is_available()):
        return TimeWrapperTime
    elif time_wrapper_type == 'cuda':
        # lazy initialize to make code runnable locally
        return get_cuda_time_wrapper()
    else:
        raise KeyError('invalid time_wrapper_type: {}'.format(time_wrapper_type))


class EasyTimer:
    """A decent timer wrapper that can be used easily.

    Example:
        wait_timer = EasyTimer()
        with wait_timer:
            func(...)
        time = wait_timer.value  # in second
    """

    def __init__(self, cuda=True):
        if torch.cuda.is_available() and cuda:
            time_wrapper_type = "cuda"
        else:
            time_wrapper_type = "time"
        self._timer = build_time_helper(wrapper_type=time_wrapper_type)
        self.value = 0.0

    def __enter__(self):
        self.value = 0.0
        self._timer.start_time()

    def __exit__(self, *args):
        self.value = self._timer.end_time()


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


def get_cuda_time_wrapper():
    class TimeWrapperCuda(TimeWrapper):
        # cls variable is initialized on loading this class
        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        # overwrite
        @classmethod
        def start_time(cls):
            torch.cuda.synchronize()
            cls.start = cls.start_record.record()

        # overwrite
        @classmethod
        def end_time(cls):
            cls.end = cls.end_record.record()
            torch.cuda.synchronize()
            return cls.start_record.elapsed_time(cls.end_record) / 1000

    return TimeWrapperCuda


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
