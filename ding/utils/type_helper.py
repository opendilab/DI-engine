from collections import namedtuple
from typing import List, Tuple, TypeVar

SequenceType = TypeVar('SequenceType', List, Tuple, namedtuple)
Tensor = TypeVar('torch.Tensor')
