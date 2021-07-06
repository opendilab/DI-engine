# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


def bimap_r(x, y, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(bimap_r(xx, y[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, bimap_r(xx, y[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y) if callback_fn is not None else None


def trimap_r(x, y, z, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(trimap_r(xx, y[i], z[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, trimap_r(xx, y[key], z[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y, z) if callback_fn is not None else None


def type_r(x):
    type_s = str(type(x))
    print(type(x))
    if isinstance(x, (list, tuple, set)):
        return {type_s: type_r(xx) for xx in x}
    elif isinstance(x, dict):
        return {type_s: type_r(xx) for xx in x.values()}
    return type_s


def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1) \
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1)) \
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1) \
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1)) \
                for key2 in x_front
            )
    return x
