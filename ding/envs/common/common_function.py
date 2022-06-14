import math
from functools import partial, lru_cache
from typing import Optional, Dict, Any

import numpy as np
import torch

from ding.compatibility import torch_ge_180
from ding.torch_utils import one_hot

num_first_one_hot = partial(one_hot, num_first=True)


def sqrt_one_hot(v: torch.Tensor, max_val: int) -> torch.Tensor:
    """
    Overview:
        Sqrt the input value ``v`` and transform it into one-hot.
    Arguments:
        - v (:obj:`torch.Tensor`): the value to be processed with `sqrt` and `one-hot`
        - max_val (:obj:`int`): the input ``v``'s estimated max value, used to calculate one-hot bit number. \
            ``v`` would be clamped by (0, max_val).
    Returns:
        - ret (:obj:`torch.Tensor`): the value processed after `sqrt` and `one-hot`
    """
    num = int(math.sqrt(max_val)) + 1
    v = v.float()
    v = torch.floor(torch.sqrt(torch.clamp(v, 0, max_val))).long()
    return one_hot(v, num)


def div_one_hot(v: torch.Tensor, max_val: int, ratio: int) -> torch.Tensor:
    """
    Overview:
        Divide the input value ``v`` by ``ratio`` and transform it into one-hot.
    Arguments:
        - v (:obj:`torch.Tensor`): the value to be processed with `divide` and `one-hot`
        - max_val (:obj:`int`): the input ``v``'s estimated max value, used to calculate one-hot bit number. \
            ``v`` would be clamped by (0, ``max_val``).
        - ratio (:obj:`int`): input ``v`` would be divided by ``ratio``
    Returns:
        - ret (:obj:`torch.Tensor`): the value processed after `divide` and `one-hot`
    """
    num = int(max_val / ratio) + 1
    v = v.float()
    v = torch.floor(torch.clamp(v, 0, max_val) / ratio).long()
    return one_hot(v, num)


def div_func(inputs: torch.Tensor, other: float, unsqueeze_dim: int = 1):
    """
    Overview:
        Divide ``inputs`` by ``other`` and unsqueeze if needed.
    Arguments:
        - inputs (:obj:`torch.Tensor`): the value to be unsqueezed and divided
        - other (:obj:`float`): input would be divided by ``other``
        - unsqueeze_dim (:obj:`int`): the dim to implement unsqueeze
    Returns:
        - ret (:obj:`torch.Tensor`): the value processed after `unsqueeze` and `divide`
    """
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)


def clip_one_hot(v: torch.Tensor, num: int) -> torch.Tensor:
    """
    Overview:
        Clamp the input ``v`` in (0, num-1) and make one-hot mapping.
    Arguments:
        - v (:obj:`torch.Tensor`): the value to be processed with `clamp` and `one-hot`
        - num (:obj:`int`): number of one-hot bits
    Returns:
        - ret (:obj:`torch.Tensor`): the value processed after `clamp` and `one-hot`
    """
    v = v.clamp(0, num - 1)
    return one_hot(v, num)


def reorder_one_hot(
        v: torch.LongTensor,
        dictionary: Dict[int, int],
        num: int,
        transform: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Overview:
        Reorder each value in input ``v`` according to reorder dict ``dictionary``, then make one-hot mapping
    Arguments:
        - v (:obj:`torch.LongTensor`): the original value to be processed with `reorder` and `one-hot`
        - dictionary (:obj:`Dict[int, int]`): a reorder lookup dict, \
            map original value to new reordered index starting from 0
        - num (:obj:`int`): number of one-hot bits
        - transform (:obj:`int`): an array to firstly transform the original action to general action
    Returns:
        - ret (:obj:`torch.Tensor`): one-hot data indicating reordered index
    """
    assert (len(v.shape) == 1)
    assert (isinstance(v, torch.Tensor))
    new_v = torch.zeros_like(v)
    for idx in range(v.shape[0]):
        if transform is None:
            val = v[idx].item()
        else:
            val = transform[v[idx].item()]
        new_v[idx] = dictionary[val]
    return one_hot(new_v, num)


def reorder_one_hot_array(
        v: torch.LongTensor, array: np.ndarray, num: int, transform: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Overview:
        Reorder each value in input ``v`` according to reorder dict ``dictionary``, then make one-hot mapping.
        The difference between this function and ``reorder_one_hot`` is
        whether the type of reorder lookup data structure is `np.ndarray` or `dict`.
    Arguments:
        - v (:obj:`torch.LongTensor`): the value to be processed with `reorder` and `one-hot`
        - array (:obj:`np.ndarray`): a reorder lookup array, map original value to new reordered index starting from 0
        - num (:obj:`int`): number of one-hot bits
        - transform (:obj:`np.ndarray`): an array to firstly transform the original action to general action
    Returns:
        - ret (:obj:`torch.Tensor`): one-hot data indicating reordered index
    """
    v = v.numpy()
    if transform is None:
        val = array[v]
    else:
        val = array[transform[v]]
    return one_hot(torch.LongTensor(val), num)


def reorder_boolean_vector(
        v: torch.LongTensor,
        dictionary: Dict[int, int],
        num: int,
        transform: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Overview:
        Reorder each value in input ``v`` to new index according to reorder dict ``dictionary``,
        then set corresponding position in return tensor to 1.
    Arguments:
        - v (:obj:`torch.LongTensor`): the value to be processed with `reorder`
        - dictionary (:obj:`Dict[int, int]`): a reorder lookup dict, \
            map original value to new reordered index starting from 0
        - num (:obj:`int`): total number of items, should equals to max index + 1
        - transform (:obj:`np.ndarray`): an array to firstly transform the original action to general action
    Returns:
        - ret (:obj:`torch.Tensor`): boolean data containing only 0 and 1, \
            indicating whether corresponding original value exists in input ``v``
    """
    ret = torch.zeros(num)
    for item in v:
        try:
            if transform is None:
                val = item.item()
            else:
                val = transform[item.item()]
            idx = dictionary[val]
        except KeyError as e:
            # print(dictionary)
            raise KeyError('{}_{}_'.format(num, e))
        ret[idx] = 1
    return ret


@lru_cache(maxsize=32)
def get_to_and(num_bits: int) -> np.ndarray:
    """
    Overview:
        Get an np.ndarray with ``num_bits`` elements, each equals to :math:`2^n` (n decreases from num_bits-1 to 0).
        Used by ``batch_binary_encode`` to make bit-wise `and`.
    Arguments:
        - num_bits (:obj:`int`): length of the generating array
    Returns:
        - to_and (:obj:`np.ndarray`): an array with ``num_bits`` elements, \
            each equals to :math:`2^n` (n decreases from num_bits-1 to 0)
    """
    return 2 ** np.arange(num_bits - 1, -1, -1).reshape([1, num_bits])


def batch_binary_encode(x: torch.Tensor, bit_num: int) -> torch.Tensor:
    """
    Overview:
        Big endian binary encode ``x`` to float tensor
    Arguments:
        - x (:obj:`torch.Tensor`): the value to be unsqueezed and divided
        - bit_num (:obj:`int`): number of bits, should satisfy :math:`2^{bit num} > max(x)`
    Example:
        >>> batch_binary_encode(torch.tensor([131,71]), 10)
        tensor([[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]])
    Returns:
        - ret (:obj:`torch.Tensor`): the binary encoded tensor, containing only `0` and `1`
    """
    x = x.numpy()
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = get_to_and(bit_num)
    return torch.FloatTensor((x & to_and).astype(bool).astype(float).reshape(xshape + [bit_num]))


def compute_denominator(x: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Compute the denominator used in ``get_postion_vector``. \
        Divide 1 at the last step, so you can use it as an multiplier.
    Arguments:
        - x (:obj:`torch.Tensor`): Input tensor, which is generated from torch.arange(0, d_model).
    Returns:
        - ret (:obj:`torch.Tensor`): Denominator result tensor.
    """
    if torch_ge_180():
        x = torch.div(x, 2, rounding_mode='trunc') * 2
    else:
        x = torch.div(x, 2) * 2
    x = torch.div(x, 64.)
    x = torch.pow(10000., x)
    x = torch.div(1., x)
    return x


def get_postion_vector(x: list) -> torch.Tensor:
    """
    Overview:
        Get position embedding used in `Transformer`, even and odd :math:`\alpha` are stored in ``POSITION_ARRAY``
    Arguments:
        - x (:obj:`list`): original position index, whose length should be 32
    Returns:
        - v (:obj:`torch.Tensor`): position embedding tensor in 64 dims
    """
    # TODO use lru_cache to optimize it
    POSITION_ARRAY = compute_denominator(torch.arange(0, 64, dtype=torch.float))  # d_model = 64
    v = torch.zeros(64, dtype=torch.float)
    x = torch.FloatTensor(x)
    v[0::2] = torch.sin(x * POSITION_ARRAY[0::2])  # even
    v[1::2] = torch.cos(x * POSITION_ARRAY[1::2])  # odd
    return v


def affine_transform(
        data: Any,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
) -> Any:
    """
    Overview:
        do affine transform for data in range [-1, 1], :math:`\alpha \times data + \beta`
    Arguments:
        - data (:obj:`Any`): the input data
        - alpha (:obj:`float`): affine transform weight
        - beta (:obj:`float`): affine transform bias
        - min_val (:obj:`float`): min value, if `min_val` and `max_val` are indicated, scale input data\
            to [min_val, max_val]
        - max_val (:obj:`float`): max value
    Returns:
        - transformed_data (:obj:`Any`): affine transformed data
    """
    data = np.clip(data, -1, 1)
    if min_val is not None:
        assert max_val is not None
        alpha = (max_val - min_val) / 2
        beta = (max_val + min_val) / 2
    assert alpha is not None
    beta = beta if beta is not None else 0.
    return data * alpha + beta
