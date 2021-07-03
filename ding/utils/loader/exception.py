from abc import ABCMeta, abstractmethod
from typing import List, Union, Tuple

INDEX_TYPING = Union[int, str]
ERROR_ITEM_TYPING = Tuple[INDEX_TYPING, Exception]
ERROR_ITEMS = List[ERROR_ITEM_TYPING]


class CompositeStructureError(ValueError, metaclass=ABCMeta):

    @property
    @abstractmethod
    def errors(self) -> ERROR_ITEMS:
        raise NotImplementedError
