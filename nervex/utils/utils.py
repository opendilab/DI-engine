from collections.abc import Sequence


def list_dict2dict_list(data):
    assert (isinstance(data, Sequence))
    if len(data) == 0:
        raise ValueError("empty data")
    if isinstance(data[0], dict):
        keys = data[0].keys()
        new_data = {k: [] for k in keys}
        for b in range(len(data)):
            for k in keys:
                new_data[k].append(data[b][k])
    elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
        new_data = list(zip(*data))
        new_data = type(data[0])(*new_data)
    else:
        raise TypeError("not support element type: {}".format(type(data[0])))
    return new_data


def dict_list2list_dict(data):
    assert (isinstance(data, dict))
    new_data = []
    for v in data.values():
        new_data.append(v)
    new_data = list(zip(*new_data))
    new_data = [{k: v for k, v in zip(data.keys(), t)} for t in new_data]
    return new_data


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


def squeeze(data):
    if isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 1:
            return data[0]
    elif isinstance(data, dict):
        if len(data) == 1:
            return data.values()[0]
    return data
