from typing import Callable, Any, List, Optional, Union
from ding.data.buffer import BufferedData


def sample_range_view(buffer_: 'Buffer', start: Optional[int] = None, end: Optional[int] = None) -> Callable:  # noqa
    assert start is not None or end is not None
    if start and start < 0:
        start = buffer_.size + start
    if end and end < 0:
        end = buffer_.size + end
    sample_range = slice(start, end)

    def _sample_range_view(action: str, chain: Callable, *args, **kwargs) -> Any:
        if action == "sample":
            return chain(*args, sample_range=sample_range)
        return chain(*args, **kwargs)

    return _sample_range_view
