from .flask_ceph_actor import FlaskCephActor, ASFlaskCephActor


class ActorCommMetaclass(type):
    def __new__(cls, name, bases, attrs):
        if '__init__' in attrs.keys():
            attrs['__init__'] = cls.enable_comm_helper(attrs['__init__'])
        return type.__new__(cls, name, bases, attrs)

    @staticmethod
    def get_comm_map():
        return {'flask_ceph': FlaskCephActor, 'as_flask_ceph': ASFlaskCephActor}

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
            comm_map = cls.get_comm_map()
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
                    setattr(instance, item, getattr(instance._comm, item))
            instance._check()
            return ret

        return wrapper
