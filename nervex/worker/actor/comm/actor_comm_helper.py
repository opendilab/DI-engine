from .base_comm_actor import BaseCommActor
from .flask_fs_actor import FlaskFileSystemActor

comm_map = {'flask_fs': FlaskFileSystemActor}


class ActorCommHelper(object):

    @classmethod
    def enable_comm_helper(cls, instance, comm_cfg):
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


def add_comm_actor(name, actor_type):
    assert isinstance(name, str)
    assert issubclass(actor_type, BaseCommActor)
    comm_map[name] = actor_type
