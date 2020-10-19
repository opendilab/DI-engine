from collections.abc import Sequence
from typing import Union


def list_dict2dict_list(data: Sequence) -> Union[list, dict, tuple]:
    if len(data) == 0:
        raise ValueError("empty data")
    if isinstance(data[0], dict):
        new_data = {k: [data[b][k] for b in range(len(data))] for k in data[0].keys()}
    elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
        new_data = type(data[0])(*list(zip(*data)))
    else:
        raise TypeError("not support element type: {}".format(type(data[0])))
    return new_data


def dict_list2list_dict(data: dict) -> list:
    new_data = [v for v in data.values()]
    new_data = [{k: v for k, v in zip(data.keys(), t)} for t in list(zip(*new_data))]
    return new_data


def override(cls: type):
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


def squeeze(data: object):
    if isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 1:
            return data[0]
    elif isinstance(data, dict):
        if len(data) == 1:
            return data.values()[0]
    return data
