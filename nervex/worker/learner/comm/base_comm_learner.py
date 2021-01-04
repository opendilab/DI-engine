import threading
from abc import ABC, abstractmethod, abstractproperty
from easydict import EasyDict

from nervex.utils import EasyTimer, import_module, get_task_uid
from nervex.policy import create_policy
from ..base_learner import BaseLearner


class BaseCommLearner(ABC):
    """
    Overview:
        Abstract baseclass for CommLearner.
    Interfaces:
        __init__, send_policy, get_data, send_learn_info
        start, close
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Initialization method
        Arguments:
            - cfg (:obj:`EasyDict`): config dict
        """
        self._cfg = cfg
        self._learner_uid = get_task_uid()
        self._timer = EasyTimer()
        self._end_flag = True

    @abstractmethod
    def send_policy(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's policy in corresponding path.
        Arguments:
            - state_dict (:obj:`dict`): state dict of the runtime policy
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self, batch_size: int) -> list:
        """
        Overview:
            Get batched data from coordinator.
        Arguments:
            - batch_size (:obj:`int`): size of one batch
        Returns:
            - stepdata (:obj:`list`): a list of train data, each element is one traj
        """
        raise NotImplementedError

    @abstractmethod
    def send_learn_info(self, learn_info: dict) -> None:
        """
        Overview:
            Send learn info to coordinator.
        Arguments:
            - learn info (:obj:`dict`): learn info in `dict` type
        """
        raise NotImplementedError

    def start(self) -> None:
        """
        Overview:
            start comm learner
        """
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close comm learner
        """
        self._end_flag = True

    @abstractproperty
    def hooks4call(self) -> list:
        """
        Returns:
            - hooks (:obj:`list`): the hooks which comm learner have, will be registered in learner as well.
        """
        raise NotImplementedError

    def _create_learner(self, task_info: dict) -> 'BaseLearner':  # noqa
        learner_cfg = EasyDict(task_info['learner_cfg'])
        learner = BaseLearner(learner_cfg)
        for item in ['get_data', 'send_policy', 'send_learn_info']:
            setattr(learner, item, getattr(self, item))
        learner.setup_dataloader()
        learner.policy = create_policy(task_info['policy'], enable_field=['learn']).learn_mode
        return learner


comm_map = {}


def register_comm_learner(name: str, learner_type: type) -> None:
    """
    Overview:
        register a new CommLearner class with its name to dict ``comm_map``
    Arguments:
        - name (:obj:`str`): name of the new CommLearner
        - learner_type (:obj:`type`): the new CommLearner class, should be subclass of BaseCommLearner
    """
    assert isinstance(name, str)
    assert issubclass(learner_type, BaseCommLearner)
    comm_map[name] = learner_type


def create_comm_learner(cfg: dict) -> BaseCommLearner:
    import_module(cfg.import_names)
    comm_learner_type = cfg.comm_learner_type
    if comm_learner_type not in comm_map.keys():
        raise KeyError("not support comm learner type: {}".format(comm_learner_type))
    else:
        return comm_map[comm_learner_type](cfg)
