from typing import Optional, Callable, List, Any, Iterable
import torch


def example_get_data_fn() -> Any:
    """
    Note: staticmethod or static function, all the operation is on CPU
    """
    # 1. read data from file or other middleware
    # 2. data post-processing(e.g.: normalization, to tensor)
    # 3. return data
    pass


class IDataLoader:

    def __next__(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Arguments:
            batch_size: sometimes, batch_size is specified by each iteration, if batch_size is None,
                use default batch_size value
        """
        # get one batch train data
        if batch_size is None:
            batch_size = self._batch_size
        data = self._get_data(batch_size)
        return self._collate_fn(data)

    def __iter__(self) -> Iterable:
        return self

    def _get_data(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        raise NotImplementedError

    def close(self) -> None:
        # release resource
        pass
