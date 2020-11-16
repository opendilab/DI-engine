import pytest
import numpy as np
import time
from nervex.utils.time_helper import build_time_helper


@pytest.mark.unittest
class TestTimeHelper:

    def test_naive(self):

        class NaiveObject(object):
            pass

        cfg = NaiveObject()
        setattr(cfg, 'common', NaiveObject())
        setattr(cfg.common, 'time_wrapper_type', 'time')
        with pytest.raises(RuntimeError):
            time_handle = build_time_helper()
        build_time_helper(cfg=None, wrapper_type="??")
        # with pytest.raises(KeyError):
        #     build_time_helper(cfg=None,wrapper_type="not_implement")
        time_handle = build_time_helper(cfg)
        time_handle = build_time_helper(wrapper_type='time')



        @time_handle.wrapper
        def func1(x):
            return x + 1

        def func2(x):
            return x + 1

        # usage 1
        ret, t = func1(3)
        assert np.isscalar(t)
        assert func1(4)[0] == func2(4)

        # usage 2
        time_handle.start_time()
        _ = func2(3)
        t = time_handle.end_time()
        assert np.isscalar(t)

        #test time_lag and restart
        time_handle.start_time()
        time.sleep(0.5)
        time_handle.start_time()
        time.sleep(1)
        t = time_handle.end_time()
        assert np.isscalar(t)
        # time_lag is bigger than 1e-3
        # assert abs(t-1) < 1e-3
        assert abs(t - 1) < 1e-2


