from collections.abc import Sequence
import torch
import numpy as np
import numbers


def to_device(item, device, ignore_keys=[]):
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_device(t, device) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_device(item[k], device)
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_dtype(item, dtype):
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, Sequence):
        return [to_dtype(t, dtype) for t in item]
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_tensor(item, dtype):
    def transform(d):
        return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_tensor(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return 'none'  # for convenience in dataloader
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype))
            return new_data
    elif item is None:
        return 'none'  # for convenience in dataloader
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def tensor_to_list(item):
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        if item.shape == (1, ):
            return item.item()
        else:
            return item.tolist()
    elif isinstance(item, list) or isinstance(item, tuple):
        return [tensor_to_list(t) for t in item]
    elif isinstance(item, dict):
        return {k: tensor_to_list(v) for k, v in item.items()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def same_shape(data):
    assert (isinstance(data, list))
    shapes = [t.shape for t in data]
    return len(set(shapes)) == 1
