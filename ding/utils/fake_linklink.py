from collections import namedtuple


class FakeClass:
    """
    Overview:
        Fake class.
    """

    def __init__(self, *args, **kwargs):
        pass


class FakeNN:
    """
    Overview:
        Fake nn class.
    """

    SyncBatchNorm2d = FakeClass


class FakeLink:
    """
    Overview:
        Fake link class.
    """

    nn = FakeNN()
    syncbnVarMode_t = namedtuple("syncbnVarMode_t", "L2")(L2=None)
    allreduceOp_t = namedtuple("allreduceOp_t", ['Sum', 'Max'])


link = FakeLink()
