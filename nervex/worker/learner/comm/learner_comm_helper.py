from typing import Callable, Any

from .base_comm_learner import BaseCommLearner
from .flask_fs_learner import FlaskFileSystemLearner

comm_map = {'flask_fs': FlaskFileSystemLearner}


class LearnerCommHelper(object):
    """
    Overview:
        A helper with only one classmethod ``enable_comm_helper``, which can initialize a comm and
        register with learner mutually to enable learner's communicating ability among multiple machines.
    Interfaces:
        enable_comm_helper
    """

    @classmethod
    def enable_comm_helper(cls, instance: 'BaseLearner', comm_cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Enable learner comm helper.

                - Init ``instance._comm`` with one comm learner in ``comm_map``.
                - Register learner's attributes (logger, id) into comm, \
                    and register comm's public attributes and hooks into learner.
                - Wrap learner's ``close`` with additional comm's ``close_service``
        Arguments:
            - instance (:obj:`BaseLearner`): the base learner which needs to register a comm
            - comm_cfg (:obj:`EasyDict`): comm config
        """

        def close_wrapper(close_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs) -> Any:
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
                    raise RuntimeError("can't set the repeated attribute({})".format(item))
                setattr(instance, item, getattr(instance._comm, item))
        instance.init_service()
        for h in instance._comm.hooks4call:
            instance.register_hook(h)

        instance.close = close_wrapper(instance.close)


def add_comm_learner(name: str, learner_type: type) -> None:
    """
    Overview:
        Add a new CommLearner class with its name to dict ``comm_map``
    Arguments:
        - name (:obj:`str`): name of the new CommLearner
        - learner_type (:obj:`type`): the new CommLearner class, should be subclass of BaseCommLearner
    """
    assert isinstance(name, str)
    assert issubclass(learner_type, BaseCommLearner)
    comm_map[name] = learner_type
