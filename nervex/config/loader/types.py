from .base import Loader, ILoaderClass


def is_type(type_: type) -> ILoaderClass:
    if isinstance(type_, type):
        return Loader(type_)
    else:
        raise TypeError('Type variable expected but {actual} found.'.format(actual=repr(type(type_).__name__)))


def to_type(type_: type) -> ILoaderClass:
    return Loader(lambda v: type_(v))


def hasprop(attr_name: str) -> ILoaderClass:
    return Loader(
        (
            lambda v: hasattr(v, attr_name),
            AttributeError('attribute {name} expected but not found'.format(name=repr(attr_name)))
        )
    )


def prop(attr_name: str) -> ILoaderClass:
    return hasprop(attr_name) & Loader(lambda v: getattr(v, attr_name))
