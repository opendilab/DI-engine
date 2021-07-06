import copy
import logging
import random
from typing import Union, Mapping, List, NamedTuple, Tuple, Callable, Optional, Any

import numpy as np
import torch


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
        >>> from ding.utils import *
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
        >>> from ding.utils import *
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
    r"""
    Overview:
        Getting the value by input, checks generically on the inputs with \
        at least ``data`` and ``name``. If ``name`` exists in ``data``, \
        get the value at ``name``; else, add ``name`` to ``default_get_set``\
        with value generated by \
        ``default_fn`` (or directly as ``default_value``) that \
        is checked by `` judge_fn`` to be legal.
    Arguments:
        - data(:obj:`dict`): Data input dictionary
        - name(:obj:`str`): Key name
        - default_value(:obj:`Optional[Any]`) = None,
        - default_fn(:obj:`Optional[Callable]`) = Value
        - judge_fn(:obj:`Optional[Callable]`) = None
    Returns:
        - ret(:obj:`list`): Splitted data
        - residual(:obj:`list`): Residule list
    """
    if name in data:
        return data[name]
    else:
        assert default_value is not None or default_fn is not None
        value = default_fn() if default_fn is not None else default_value
        if judge_fn:
            assert judge_fn(value), "defalut value({}) is not accepted by judge_fn".format(type(value))
        if name not in default_get_set:
            logging.warning("{} use default value {}".format(name, value))
            default_get_set.add(name)
        return value


def list_split(data: list, step: int) -> List[list]:
    r"""
    Overview:
        Split list of data by step.
    Arguments:
        - data(:obj:`list`): List of data for spliting
        - step(:obj:`int`): Number of step for spliting
    Returns:
        - ret(:obj:`list`): List of splitted data.
        - residual(:obj:`list`): Residule list. This value is ``None`` when  ``data`` divides ``steps``.
    Example:
        >>> list_split([1,2,3,4],2)
        ([[1, 2], [3, 4]], None)
        >>> list_split([1,2,3,4],3)
        ([[1, 2, 3]], [4])
    """
    if len(data) < step:
        return [], data
    ret = []
    divide_num = len(data) // step
    for i in range(divide_num):
        start, end = i * step, (i + 1) * step
        ret.append(data[start:end])
    if divide_num * step < len(data):
        residual = data[divide_num * step:]
    else:
        residual = None
    return ret, residual


def error_wrapper(fn, default_ret, warning_msg=""):
    r"""
    Overview:
        wrap the function, so that any Exception in the function will be catched and return the default_ret
    Arguments:
        - fn (:obj:`Callable`): the function to be wraped
        - default_ret (:obj:`obj`): the default return when an Exception occurred in the function
    Returns:
        - wrapper (:obj:`Callable`): the wrapped function
    Examples:
        >>> # Used to checkfor Fakelink (Refer to utils.linklink_dist_helper.py)
        >>> def get_rank():  # Get the rank of linklink model, return 0 if use FakeLink.
        >>>    if is_fake_link:
        >>>        return 0
        >>>    return error_wrapper(link.get_rank, 0)()
    """

    def wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
        except Exception as e:
            ret = default_ret
            if warning_msg != "":
                logging.warning(warning_msg, "\ndefault_ret = {}\terror = {}".format(default_ret, e))
        return ret

    return wrapper


class LimitedSpaceContainer:
    r"""
    Overview:
        A space simulator.
    Interface:
        ``__init__``, ``get_residual_space``, ``release_space``
    """

    def __init__(self, min_val: int, max_val: int) -> None:
        """
        Overview:
            Set ``min_val`` and ``max_val`` of the container, also set ``cur`` to ``min_val`` for initialization.
        Arguments:
            - min_val (:obj:`int`): Min volume of the container, usually 0.
            - max_val (:obj:`int`): Max volume of the container.
        """
        self.min_val = min_val
        self.max_val = max_val
        assert (max_val >= min_val)
        self.cur = self.min_val

    def get_residual_space(self) -> int:
        """
        Overview:
            Get all residual pieces of space. Set ``cur`` to ``max_val``
        Arguments:
            - ret (:obj:`int`): Residual space, calculated by ``max_val`` - ``cur``.
        """
        ret = self.max_val - self.cur
        self.cur = self.max_val
        return ret

    def acquire_space(self) -> bool:
        """
        Overview:
            Try to get one pice of space. If there is one, return True; Otherwise return False.
        Returns:
            - flag (:obj:`bool`): Whether there is any piece of residual space.
        """
        if self.cur < self.max_val:
            self.cur += 1
            return True
        else:
            return False

    def release_space(self) -> None:
        """
        Overview:
            Release only one piece of space. Decrement ``cur``, but ensure it won't be negative.
        """
        self.cur = max(self.min_val, self.cur - 1)

    def increase_space(self) -> None:
        """
        Overview:
            Increase one piece in space. Increment ``max_val``.
        """
        self.max_val += 1

    def decrease_space(self) -> None:
        """
        Overview:
            Decrease one piece in space. Decrement ``max_val``.
        """
        self.max_val -= 1


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        Merge two dicts by calling ``deep_update``
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - merged_dict (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])
    return merged


def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: bool = False,
    whitelist: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None
):
    """
    Overview:
        Update original dict with values from new_dict recursively.
    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (:obj:`Optional[List[str]]`):
            List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(:obj:`Optional[List[str]]`):
            List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def flatten_dict(data: dict, delimiter: str = "/") -> dict:
    """
    Overview:
        Flatten the dict, see example
    Arguments:
        - data (:obj:`dict`): Original nested dict
        - delimiter (str): Delimiter of the keys of the new dict
    Returns:
        - data (:obj:`dict`): Flattened nested dict
    Example:
        >>> a
        {'a': {'b': 100}}
        >>> flatten_dict(a)
        {'a/b': 100}
    """
    data = copy.deepcopy(data)
    while any(isinstance(v, dict) for v in data.values()):
        remove = []
        add = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        data.update(add)
        for k in remove:
            del data[k]
    return data


def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
    """
    Overview:
        Side effect function to set seed for ``random``, ``numpy random``, and ``torch's manual seed``.\
        This is usaually used in entry scipt in the section of setting random seed for all package and instance
    Argument:
        - seed(:obj:`int`): Set seed
        - use_cuda(:obj:`bool`) Whether use cude
    Examples:
        >>> # ../entry/xxxenv_xxxpolicy_main.py
        >>> ...
        # Set random seed for all package and instance
        >>> collector_env.seed(seed)
        >>> evaluator_env.seed(seed, dynamic_seed=False)
        >>> set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
        >>> ...
        # Set up RL Policy, etc.
        >>> ...

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
