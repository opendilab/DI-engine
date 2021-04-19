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

    def __init__(
            self,
            get_data_fn: Callable,
            batch_size: int,
            collate_fn: Optional[Callable] = None,
            num_workers: int = 0
    ) -> None:
        """
        Arguments:
            batch_size: the number of samples in one iteration
            collate_fn: the function of stack all the data into a batch
            num_workers: the number of worker process
        """
        self._batch_size = batch_size
        # Every time get_data_fn is called, returns a training sample
        self._get_data_fn = get_data_fn
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        # create resource in init method

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

    def _get_data(self, batch_size: int) -> List[torch.Tensor]:
        """
        Overview:
            use get_data_fn to get data
        """
        pass

    def close(self) -> None:
        # release resource
        pass

    def __del__(self) -> None:
        self.close()
