from typing import Callable, Any, List, TYPE_CHECKING
if TYPE_CHECKING:
    from ding.data.buffer.buffer import Buffer


def staleness_check(buffer_: 'Buffer', max_staleness: int = float("inf")) -> Callable:
    """
    Overview:
        This middleware aims to check staleness before each sample operation,
        staleness = train_iter_sample_data - train_iter_data_collected, means how old/off-policy the data is,
        If data's staleness is greater(>) than max_staleness, this data will be removed from buffer as soon as possible.
    Arguments:
        - max_staleness (:obj:`int`): The maximum legal span between the time of collecting and time of sampling.
    """

    def push(next: Callable, data: Any, *args, **kwargs) -> Any:
        assert 'meta' in kwargs and 'train_iter_data_collected' in kwargs[
            'meta'], "staleness_check middleware must push data with meta={'train_iter_data_collected': <iter>}"
        return next(data, *args, **kwargs)

    def sample(next: Callable, train_iter_sample_data: int, *args, **kwargs) -> List[Any]:
        delete_index = []
        for i, item in enumerate(buffer_.storage):
            index, meta = item.index, item.meta
            staleness = train_iter_sample_data - meta['train_iter_data_collected']
            meta['staleness'] = staleness
            if staleness > max_staleness:
                delete_index.append(index)
        for index in delete_index:
            buffer_.delete(index)
        data = next(*args, **kwargs)
        return data

    def _staleness_check(action: str, next: Callable, *args, **kwargs) -> Any:
        if action == "push":
            return push(next, *args, **kwargs)
        elif action == "sample":
            return sample(next, *args, **kwargs)
        return next(*args, **kwargs)

    return _staleness_check
