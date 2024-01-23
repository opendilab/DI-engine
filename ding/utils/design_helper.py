from abc import ABCMeta


# ABCMeta is a subclass of type, extending ABCMeta makes this metaclass is compatible with some classes
# which extends ABC
class SingletonMetaclass(ABCMeta):
    """
    Overview:
        Returns the given type instance in input class
    Interfaces:
        ``__call__``
    """
    instances = {}

    def __call__(cls: type, *args, **kwargs) -> object:
        """
        Overview:
            Returns the given type instance in input class
        """

        if cls not in SingletonMetaclass.instances:
            SingletonMetaclass.instances[cls] = super(SingletonMetaclass, cls).__call__(*args, **kwargs)
            cls.instance = SingletonMetaclass.instances[cls]
        return SingletonMetaclass.instances[cls]
