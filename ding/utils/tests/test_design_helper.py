import random

import pytest

from ding.utils import SingletonMetaclass


@pytest.mark.unittest
def test_singleton():
    global count
    count = 0

    class A(object, metaclass=SingletonMetaclass):

        def __init__(self, t):
            self.t = t
            self.p = random.randint(0, 10)
            global count
            count += 1

    obj = [A(i) for i in range(3)]
    assert count == 1
    assert all([o.t == 0 for o in obj])
    assert all([o.p == obj[0].p for o in obj])
    assert all([id(o) == id(obj[0]) for o in obj])
    assert id(A.instance) == id(obj[0])

    # subclass test
    class B(A):
        pass

    obj = [B(i) for i in range(3, 6)]
    assert count == 2
    assert all([o.t == 3 for o in obj])
    assert all([o.p == obj[0].p for o in obj])
    assert all([id(o) == id(obj[0]) for o in obj])
    assert id(B.instance) == id(obj[0])
