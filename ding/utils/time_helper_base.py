class TimeWrapper(object):
    r"""
    Overview:
        Abstract class method that defines ``TimeWrapper`` class

    Interface:
        ``wrapper``, ``start_time``, ``end_time``
    """

    @classmethod
    def wrapper(cls, fn):
        r"""
        Overview:
            Classmethod wrapper, wrap a function and automatically return its running time

        - fn (:obj:`function`): The function to be wrap and timed
        """

        def time_func(*args, **kwargs):
            cls.start_time()
            ret = fn(*args, **kwargs)
            t = cls.end_time()
            return ret, t

        return time_func

    @classmethod
    def start_time(cls):
        r"""
        Overview:
            Abstract classmethod, start timing
        """
        raise NotImplementedError

    @classmethod
    def end_time(cls):
        r"""
        Overview:
            Abstract classmethod, stop timing
        """
        raise NotImplementedError
