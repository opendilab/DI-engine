import math
from functools import partial, lru_cache

import numpy as np
import torch

from nervex.torch_utils import one_hot

num_first_one_hot = partial(one_hot, num_first=True)


def sqrt_one_hot(v, max_val):
    num = int(math.sqrt(max_val)) + 1
    v = v.float()
    v = torch.floor(torch.sqrt(torch.clamp(v, 0, max_val))).long()
    return one_hot(v, num)


def div_one_hot(v, max_val, ratio):
    num = int(max_val / ratio) + 1
    v = v.float()
    v = torch.floor(torch.clamp(v, 0, max_val) / ratio).long()
    return one_hot(v, num)


def reorder_one_hot(v, dictionary, num, transform=None):
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


def reorder_one_hot_array(v, array, num, transform=None):
    v = v.numpy()
    if transform is None:
        val = array[v]
    else:
        val = array[transform[v]]
    return one_hot(torch.LongTensor(val), num)


def div_func(inputs, other, unsqueeze_dim=1):
    inputs = inputs.float()
    if unsqueeze_dim is not None:
        inputs = inputs.unsqueeze(unsqueeze_dim)
    return torch.div(inputs, other)


@lru_cache(maxsize=32)
def get_to_and(num_bits):
    return 2**np.arange(num_bits - 1, -1, -1).reshape([1, num_bits])


def batch_binary_encode(x, bit_num):
    # Big endian binary encode to float tensor
    # Example: >>> batch_binary_encode(torch.tensor([131,71]), 10)
    # tensor([[0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
    #         [0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]])
    x = x.numpy()
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = get_to_and(bit_num)
    return torch.FloatTensor((x & to_and).astype(bool).astype(float).reshape(xshape + [bit_num]))


def reorder_boolean_vector(v, dictionary, num, transform=None):
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


def clip_one_hot(v, num):
    v = v.clamp(0, num - 1)
    return one_hot(v, num)


def compute_denominator(x):
    x = x // 2 * 2
    x = torch.div(x, 64.)
    x = torch.pow(10000., x)
    x = torch.div(1., x)
    return x


POSITION_ARRAY = compute_denominator(torch.arange(0, 64, dtype=torch.float))


def get_postion_vector(x):
    v = torch.zeros(64, dtype=torch.float)
    x = torch.FloatTensor(x)
    v[0::2] = torch.sin(x * POSITION_ARRAY[0::2])
    v[1::2] = torch.cos(x * POSITION_ARRAY[1::2])
    return v
