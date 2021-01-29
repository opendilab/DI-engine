from abc import ABCMeta


class CompositeStructureError(ValueError, metaclass=ABCMeta):
    pass
