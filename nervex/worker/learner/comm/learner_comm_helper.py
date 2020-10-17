from .base_comm_learner import BaseCommLearner
from .flask_fs_learner import FlaskFileSystemLearner

comm_map = {'flask_fs': FlaskFileSystemLearner}


class LearnerCommHelper(object):

    @classmethod
    def enable_comm_helper(cls, instance, comm_cfg):

        def close_wrapper(close_fn):

            def wrapper(*args, **kwargs):
                ret = close_fn(*args, **kwargs)
                instance.close_service()
                return ret

            return wrapper

        comm_type = comm_cfg['type']
        if comm_type not in comm_map:
            raise KeyError(comm_type)
        else:
            instance._comm = comm_map[comm_type](comm_cfg)

        # instance -> comm
        instance._comm._logger = instance._logger
        instance._comm._learner_uid = instance._learner_uid
        instance._comm._learner_worker_uid = instance._learner_worker_uid

        # comm -> instance
        for item in dir(instance._comm):
            if not item.startswith('_'):  # only public method and variable
                if hasattr(instance, item):
                    raise RuntimeError("can't set the repeat attribute({})".format(item))
                setattr(instance, item, getattr(instance._comm, item))
        instance.init_service()
        for h in instance._comm.hooks4call:
            instance.register_hook(h)

        instance.close = close_wrapper(instance.close)


def add_comm_learner(name, learner_type):
    assert isinstance(name, str)
    assert issubclass(learner_type, BaseCommLearner)
    comm_map[name] = learner_type
