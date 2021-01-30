from .base import Loader


def hasprop(attr_name: str):
    return Loader(
        (
            lambda v: hasattr(v, attr_name),
            AttributeError('attribute {name} expected but not found'.format(name=repr(attr_name)))
        )
    )


def prop(attr_name: str):
    return hasprop(attr_name) & Loader(lambda v: getattr(v, attr_name))
