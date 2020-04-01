from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
from datetime import datetime

import numpy as np
import torch
from absl import flags


def deepcopy(data):
    if data is None:
        new_data = data
    elif isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_data[k] = deepcopy(v)
    elif isinstance(data, list) or isinstance(data, tuple):
        new_data = []
        for item in data:
            new_data.append(deepcopy(item))
    elif isinstance(data, torch.Tensor):
        new_data = data.clone()
    elif isinstance(data, np.ndarray):
        new_data = np.copy(data)
    elif isinstance(data, str) or isinstance(data, numbers.Integral):
        new_data = data
    else:
        raise TypeError("invalid data type:{}".format(type(data)))
    return new_data


def list_dict2dict_list(data):
    assert (isinstance(data, list))
    if len(data) == 0:
        raise ValueError("empty data")
    keys = data[0].keys()
    new_data = {k: [] for k in keys}
    for b in range(len(data)):
        for k in keys:
            new_data[k].append(data[b][k])
    return new_data


def dict_list2list_dict(data):
    assert (isinstance(data, dict))
    new_data = []
    for v in data.values():
        new_data.append(v)
    new_data = list(zip(*new_data))
    new_data = [{k: v for k, v in zip(data.keys(), t)} for t in new_data]
    return new_data


def merge_two_dicts(x, y):
    """Merge two dicts into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def override(cls):
    """Annotation for documenting method overrides.

    Arguments:
        cls (type): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    """
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override
