from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union


class Storage:
    """
    Storage is an abstraction of device storage, third-party services or data structures,
    For example, memory queue, sum-tree, redis, or di-store.
    """

    @abstractmethod
    def append(self, data: Any, meta: Optional[dict] = None) -> None:
        """
        Overview:
            Push data and it's meta information in buffer.
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
            self,
            size: int,
            replace: bool = False,
            range: Optional[slice] = None,
            return_index: bool = False,
            return_meta: bool = False
    ) -> List[Union[Any, Tuple[Any, str], Tuple[Any, str, dict]]]:
        """
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - range (:obj:`slice`): Range slice.
            - return_index (:obj:`bool`): Transform the return value to (data, index),
            - return_meta (:obj:`bool`): Transform the return value to (data, meta),
                or (data, index, meta) if return_index is true.
        Returns:
            - sample_data (:obj:`list`): A list of data with length ``size``.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, index: str, data: Any, meta: Optional[Any] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of data.
            - data (:obj:`any`): Pure data.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, index: str) -> bool:
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
