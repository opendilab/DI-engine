import random
from typing import Iterable


def random_port(excludes: Iterable[int] = None) -> int:
    return random.choice(list(set(range(10000, 20000)) - set(excludes or [])))


def random_channel() -> int:
    return random.randint(1000, (1 << 31) - 1)
