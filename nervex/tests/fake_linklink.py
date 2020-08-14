from collections import namedtuple

import torch


class FakeClass:
    def __init__(self, *args, **kwargs):
        pass


class FakeNN:
    SyncBatchNorm2d = FakeClass

    # def __getattr__(self, item):
    #     result = getattr(torch.nn, item)


class FakeLink:
    nn = FakeNN()
    syncbnVarMode_t = namedtuple("syncbnVarMode_t", "L2")(L2=None)
    allreduceOp_t = namedtuple("allreduceOp_t", ['Sum', 'Max'])


link = FakeLink()
