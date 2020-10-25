from typing import Union, Mapping, List, NamedTuple, Tuple, Callable


def lists_to_dicts(
    data: Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]]
) -> Union[Mapping[object, object], NamedTuple]:
    """
    Transform a list of dicts to a dict of lists.

    Args:
        data (Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]]):
            A dict of lists need to be transformed

    Returns:
        Union[Mapping[object, object], NamedTuple]: A list of dicts as a result

    Example:
        >>> from nervex.utils import *
        >>> lists_to_dicts([{1: 1, 10: 3}, {1: 2, 10: 4}])
        {1: [1, 2], 10: [3, 4]}
    """
    if len(data) == 0:
        raise ValueError("empty data")
    if isinstance(data[0], dict):
        new_data = {k: [data[b][k] for b in range(len(data))] for k in data[0].keys()}
    elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
        new_data = type(data[0])(*list(zip(*data)))
    else:
        raise TypeError("not support element type: {}".format(type(data[0])))
    return new_data


def dicts_to_lists(data: Mapping[object, List[object]]) -> List[Mapping[object, object]]:
    """
    Transform a dict of lists to a list of dicts.

    Args:
        data (Mapping[object, list]): A list of dicts need to be transformed

    Returns:
        List[Mapping[object, object]]: A dict of lists as a result

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
    """Annotation for documenting method overrides.

    Arguments:
        cls (type): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    """

    def check_override(method: Callable) -> Callable:
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def squeeze(data: object) -> object:
    """
    Squeeze data from tuple, list or dict to single object
    """
    if isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 1:
            return data[0]
    elif isinstance(data, dict):
        if len(data) == 1:
            return data.values()[0]
    return data
