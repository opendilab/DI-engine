from abc import ABCMeta
from .base_comm_learner import BaseCommLearner
from .flask_fs_learner import FlaskFileSystemLearner

comm_map = {'flask_fs': FlaskFileSystemLearner}


class LearnerCommMetaclass(ABCMeta):
    def __new__(cls, name, bases, attrs):
        # TODO
        if '__init__' in attrs.keys():
            attrs['__init__'], attrs['close'] = cls.enable_comm_helper(attrs['__init__'], attrs['close'])
        return type.__new__(cls, name, bases, attrs)

    @classmethod
    def enable_comm_helper(cls, init_fn, close_fn):
        def init_wrapper(*args, **kwargs):
            if 'comm_cfg' in kwargs.keys():
                comm_cfg = kwargs.pop('comm_cfg')
            else:
                print("Single Machine Learner has launched")
                return init_fn(*args, **kwargs)

            ret = init_fn(*args, **kwargs)
            instance = args[0]
            comm_type = comm_cfg['type']
            if comm_type not in comm_map:
                raise KeyError(comm_type)
            else:
                instance._comm = comm_map[comm_type](comm_cfg)

            # instance -> comm
            instance._comm._logger = instance._logger
            instance._comm._learner_uid = instance._learner_uid

            # comm -> instance
            for item in dir(instance._comm):
                if not item.startswith('_'):  # only public method and variable
                    if hasattr(instance, item):
                        raise RuntimeError("can't set the repeat attribute({})".format(item))
                    setattr(instance, item, getattr(instance._comm, item))
            instance.init_service()
            for h in instance._comm.hooks4call:
                instance.register_hook(h)
            return ret

        def close_wrapper(*args, **kwargs):
            ret = close_fn(*args, **kwargs)
            instance = args[0]
            instance.close_service()
            return ret

        return init_wrapper, close_wrapper


def add_comm_learner(name, learner_type):
    assert isinstance(name, str)
    assert issubclass(learner_type, BaseCommLearner)
    comm_map[name] = learner_type
