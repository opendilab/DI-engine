"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
from typing import Union, Mapping, List, NamedTuple, Tuple, Callable, Optional, Any
import copy
import warnings


def lists_to_dicts(
        data: Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]],
        recursive: bool = False,
) -> Union[Mapping[object, object], NamedTuple]:
    r"""
    Overview:
        Transform a list of dicts to a dict of lists.

    Arguments:
        - data (:obj:`Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]]`):
            A dict of lists need to be transformed
        - recursive (:obj:`bool`): whether recursively deals with dict element

    Returns:
        - newdata (:obj:`Union[Mapping[object, object], NamedTuple]`): A list of dicts as a result

    Example:
        >>> from nervex.utils import *
        >>> lists_to_dicts([{1: 1, 10: 3}, {1: 2, 10: 4}])
        {1: [1, 2], 10: [3, 4]}
    """
    if len(data) == 0:
        raise ValueError("empty data")
    if isinstance(data[0], dict):
        if recursive:
            new_data = {}
            for k in data[0].keys():
                if isinstance(data[0][k], dict):
                    tmp = [data[b][k] for b in range(len(data))]
                    new_data[k] = lists_to_dicts(tmp)
                else:
                    new_data[k] = [data[b][k] for b in range(len(data))]
        else:
            new_data = {k: [data[b][k] for b in range(len(data))] for k in data[0].keys()}
    elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
        new_data = type(data[0])(*list(zip(*data)))
    else:
        raise TypeError("not support element type: {}".format(type(data[0])))
    return new_data


def dicts_to_lists(data: Mapping[object, List[object]]) -> List[Mapping[object, object]]:
    r"""
    Overview:
        Transform a dict of lists to a list of dicts.

    Arguments:
        - data (:obj:`Mapping[object, list]`): A list of dicts need to be transformed

    Returns:
        - newdata (:obj:`List[Mapping[object, object]]`): A dict of lists as a result

    Example:
        >>> from nervex.utils import *
        >>> dicts_to_lists({1: [1, 2], 10: [3, 4]})
        [{1: 1, 10: 3}, {1: 2, 10: 4}]
    """
    new_data = [v for v in data.values()]
    new_data = [{k: v for k, v in zip(data.keys(), t)} for t in list(zip(*new_data))]
    return new_data


def override(cls: type) -> Callable[[
        Callable,
], Callable]:
    """
    Overview:
        Annotation for documenting method overrides.

    Arguments:
        - cls (:obj:`type`): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    """

    def check_override(method: Callable) -> Callable:
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def squeeze(data: object) -> object:
    """
    Overview:
        Squeeze data from tuple, list or dict to single object
    Example:
        >>> a = (4, )
        >>> a = squeeze(a)
        >>> print(a)
        >>> 4
    """
    if isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 1:
            return data[0]
        else:
            return tuple(data)
    elif isinstance(data, dict):
        if len(data) == 1:
            return list(data.values())[0]
    return data


default_get_set = set()


def default_get(
        data: dict,
        name: str,
        default_value: Optional[Any] = None,
        default_fn: Optional[Callable] = None,
        judge_fn: Optional[Callable] = None
) -> Any:
    if name in data:
        return data[name]
    else:
        assert default_value is not None or default_fn is not None
        value = default_fn() if default_fn is not None else default_value
        if judge_fn:
            assert judge_fn(value), "defalut value({}) is not accepted by judge_fn".format(type(value))
        if name not in default_get_set:
            warnings.warn("{} use default value {}".format(name, value))
            default_get_set.add(name)
        return value


def list_split(data: list, step: int) -> List[list]:
    if len(data) < step:
        warnings.warn("list_split data real length is shorted than the given step({})".format(step))
        return []
    ret = []
    divide_num = len(data) // step
    for i in range(divide_num):
        start, end = i * step, (i + 1) * step
        ret.append(data[start:end])
    if divide_num * step < len(data):
        ret.append(copy.deepcopy(data[-step:]))
    return ret


def error_wrapper(fn, default_ret, warning_msg="[WARNING]: call linklink error, return default_ret."):
    r"""
    Overview:
        wrap the function, so that any Exception in the function will be catched and return the default_ret
    Arguments:
        - fn (:obj:`Callable`): the function to be wraped
        - default_ret (:obj:`obj`): the default return when an Exception occured in the function
    Returns:
        - wrapper (:obj:`Callable`): the wrapped function
    """

    def wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
        except Exception:
            ret = default_ret
            print(warning_msg, "\ndefault_ret = {}".format(default_ret))
        return ret

    return wrapper
