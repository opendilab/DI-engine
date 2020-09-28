from abc import ABCMeta
from .base_comm_actor import BaseCommActor
from .flask_fs_actor import FlaskFileSystemActor

comm_map = {'flask_fs': FlaskFileSystemActor}


class ActorCommMetaclass(ABCMeta):
    def __new__(cls, name, bases, attrs):
        if '__init__' in attrs.keys():
            attrs['__init__'] = cls.enable_comm_helper(attrs['__init__'])
        return type.__new__(cls, name, bases, attrs)

    @classmethod
    def enable_comm_helper(cls, fn):
        def wrapper(*args, **kwargs):
            if 'comm_cfg' in kwargs.keys():
                comm_cfg = kwargs.pop('comm_cfg')
            else:
                print('[WARNING]: use default single machine communication strategy', kwargs)
                # TODO single machine actor
                raise NotImplementedError

            ret = fn(*args, **kwargs)
            instance = args[0]
            comm_type = comm_cfg['type']
            if comm_type not in comm_map.keys():
                raise KeyError(comm_type)
            else:
                instance._comm = comm_map[comm_type](comm_cfg)

            # instance -> comm
            instance._comm.actor_uid = instance._actor_uid
            instance._comm._logger = instance._logger

            # comm -> instance
            for item in dir(instance._comm):
                if not item.startswith('_'):  # only public method and variable
                    if hasattr(instance, item):
                        raise RuntimeError("can't set the repeat attribute({})".format(item))
                    setattr(instance, item, getattr(instance._comm, item))
            instance._check()
            return ret

        return wrapper


def add_comm_actor(name, actor_type):
    assert isinstance(name, str)
    assert issubclass(actor_type, BaseCommActor)
    comm_map[name] = actor_type
