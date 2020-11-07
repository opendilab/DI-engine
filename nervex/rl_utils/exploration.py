import math
from typing import Callable


def epsilon_greedy(start: float, end: float, decay: int, type_: str = 'exp') -> Callable:
    assert type_ in ['linear', 'exp'], type_
    if type_ == 'exp':
        return lambda x: (start - end) * math.exp(-1 * x / decay) + end
    elif type_ == 'linear':
        assert start == 1.0, start

        def eps_fn(x):
            if x >= decay:
                return end
            else:
                return (start - x / decay) * (start - end)

        return eps_fn
