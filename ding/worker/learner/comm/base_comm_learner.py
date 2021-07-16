from abc import ABC, abstractmethod, abstractproperty
from easydict import EasyDict

from ding.utils import EasyTimer, import_module, get_task_uid, dist_init, dist_finalize, COMM_LEARNER_REGISTRY
from ding.policy import create_policy
from ding.worker.learner import create_learner


class BaseCommLearner(ABC):
    """
    Overview:
        Abstract baseclass for CommLearner.
    Interfaces:
        __init__, send_policy, get_data, send_learn_info， start, close
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
        """
        self._cfg = cfg
        self._learner_uid = get_task_uid()
        self._timer = EasyTimer()
        if cfg.multi_gpu:
            self._rank, self._world_size = dist_init()
        else:
            self._rank, self._world_size = 0, 1
        self._multi_gpu = cfg.multi_gpu
        self._end_flag = True

    @abstractmethod
    def send_policy(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's policy in corresponding path.
            Will be registered in base learner.
        Arguments:
            - state_dict (:obj:`dict`): State dict of the runtime policy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self, batch_size: int) -> list:
        """
        Overview:
            Get batched meta data from coordinator.
            Will be registered in base learner.
        Arguments:
            - batch_size (:obj:`int`): Batch size.
        Returns:
            - stepdata (:obj:`list`): A list of training data, each element is one trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def send_learn_info(self, learn_info: dict) -> None:
        """
        Overview:
            Send learn info to coordinator.
            Will be registered in base learner.
        Arguments:
            - learn_info (:obj:`dict`): Learn info in dict type.
        """
        raise NotImplementedError

    def start(self) -> None:
        """
        Overview:
            Start comm learner.
        """
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close comm learner.
        """
        self._end_flag = True
        if self._multi_gpu:
            dist_finalize()

    @abstractproperty
    def hooks4call(self) -> list:
        """
        Returns:
            - hooks (:obj:`list`): The hooks which comm learner has. Will be registered in learner as well.
        """
        raise NotImplementedError

    def _create_learner(self, task_info: dict) -> 'BaseLearner':  # noqa
        """
        Overview:
            Receive ``task_info`` passed from coordinator and create a learner.
        Arguments:
            - task_info (:obj:`dict`): Task info dict from coordinator. Should be like \
                {"learner_cfg": xxx, "policy": xxx}.
        Returns:
            - learner (:obj:`BaseLearner`): Created base learner.

        .. note::
            Three methods('get_data', 'send_policy', 'send_learn_info'), dataloader and policy are set.
            The reason why they are set here rather than base learner is that, they highly depend on the specific task.
            Only after task info is passed from coordinator to comm learner through learner slave, can they be
            clarified and initialized.
        """
        # Prepare learner config and instantiate a learner object.
        learner_cfg = EasyDict(task_info['learner_cfg'])
        learner = create_learner(learner_cfg, dist_info=[self._rank, self._world_size], exp_name=learner_cfg.exp_name)
        # Set 3 methods and dataloader in created learner that are necessary in parallel setting.
        for item in ['get_data', 'send_policy', 'send_learn_info']:
            setattr(learner, item, getattr(self, item))
        # Set policy in created learner.
        policy_cfg = task_info['policy']
        learner.policy = create_policy(policy_cfg, enable_field=['learn']).learn_mode
        learner.setup_dataloader()
        return learner


def create_comm_learner(cfg: EasyDict) -> BaseCommLearner:
    """
    Overview:
        Given the key(comm_learner_name), create a new comm learner instance if in comm_map's values,
        or raise an KeyError. In other words, a derived comm learner must first register,
        then can call ``create_comm_learner`` to get the instance.
    Arguments:
        - cfg (:obj:`dict`): Learner config. Necessary keys: [import_names, comm_learner_type].
    Returns:
        - learner (:obj:`BaseCommLearner`): The created new comm learner, should be an instance of one of \
            comm_map's values.
    """
    import_module(cfg.get('import_names', []))
    return COMM_LEARNER_REGISTRY.build(cfg.type, cfg=cfg)
