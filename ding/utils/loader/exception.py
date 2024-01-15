from abc import ABCMeta, abstractmethod
from typing import List, Union, Tuple

INDEX_TYPING = Union[int, str]
ERROR_ITEM_TYPING = Tuple[INDEX_TYPING, Exception]
ERROR_ITEMS = List[ERROR_ITEM_TYPING]


class CompositeStructureError(ValueError, metaclass=ABCMeta):
    """
    Overview:
        Composite structure error.
    Interfaces:
        ``__init__``, ``errors``
    Properties:
        ``errors``
    """

    @property
    @abstractmethod
    def errors(self) -> ERROR_ITEMS:
        """
        Overview:
            Get the errors.
        """

        raise NotImplementedError
