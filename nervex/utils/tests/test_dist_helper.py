import pytest

from nervex.utils.dist_helper import dist_init, dist_finalize, distributed_mode


@pytest.mark.unittest
class TestDist():

    #     def test_dist_init_finalize(self):
    #         dist_init()
    #         dist_finalize()

    #     def test_distributed_mode(self):
    #         def fn1(x):
    #             return x+1

    #         fn2 = distributed_mode(fn1)

    #         assert fn2(1) == 2
    def test_dist(self):
        pass
