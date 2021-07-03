from collections import namedtuple


class FakeClass:

    def __init__(self, *args, **kwargs):
        pass


class FakeNN:
    SyncBatchNorm2d = FakeClass


class FakeLink:
    nn = FakeNN()
    syncbnVarMode_t = namedtuple("syncbnVarMode_t", "L2")(L2=None)
    allreduceOp_t = namedtuple("allreduceOp_t", ['Sum', 'Max'])


link = FakeLink()
