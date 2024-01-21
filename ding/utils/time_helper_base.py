class TimeWrapper(object):
    """
    Overview:
        Abstract class method that defines ``TimeWrapper`` class

    Interfaces:
        ``wrapper``, ``start_time``, ``end_time``
    """

    @classmethod
    def wrapper(cls, fn):
        """
        Overview:
            Classmethod wrapper, wrap a function and automatically return its running time
        Arguments:
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
        """
        Overview:
            Abstract classmethod, start timing
        """
        raise NotImplementedError

    @classmethod
    def end_time(cls):
        """
        Overview:
            Abstract classmethod, stop timing
        """
        raise NotImplementedError
