import threading
from abc import ABC, abstractmethod, abstractproperty

from nervex.utils import EasyTimer


class BaseCommLearner(ABC):
    """
    Overview:
        Abstract baseclass for CommLearner.
    Interfaces:
        __init__, register_learner, send_agent, get_data, send_train_info, start_heartbeats_thread
        init_service, close_service,
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
        self._learner_uid = None  # str(os.environ.get('SLURM_JOB_ID'))
        self._active_flag = False
        self._timer = EasyTimer()

    @abstractmethod
    def register_learner(self) -> None:
        """
        Overview:
            Register learner's info in coordinator, called by ``self.init_service``.
        """
        raise NotImplementedError

    @abstractmethod
    def send_agent(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's agent in corresponding path.
        Arguments:
            - state_dict (:obj:`dict`): state dict of the runtime agent
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
    def send_train_info(self, train_info: dict) -> None:
        """
        Overview:
            Send train info to coordinator.
        Arguments:
            - train info (:obj:`dict`): train info in `dict` type
        """
        raise NotImplementedError

    def start_heartbeats_thread(self) -> None:
        """
        Overview:
            Start ``_send_learner_heartbeats`` as a daemon thread to continuously send learner heartbeats,
            called by ``self.init_service``
        """
        check_send_learner_heartbeats_thread = threading.Thread(target=self._send_learner_heartbeats)
        check_send_learner_heartbeats_thread.daemon = True
        check_send_learner_heartbeats_thread.start()
        self._logger.info("Learner({}) send heartbeat thread start...".format(self._learner_uid))

    def init_service(self) -> None:
        """
        Overview:
            Initialize comm service, including ``register_learner``, setting ``_active_flag`` to True, and
            ``start_heartbeats_thread``
        """
        self.register_learner()
        self._active_flag = True
        self.start_heartbeats_thread()

    def close_service(self) -> None:
        """
        Overview:
            Close comm service, including setting ``_active_flag`` to False
        """
        self._active_flag = False

    # ************************** thread *********************************
    @abstractmethod
    def _send_learner_heartbeats(self) -> None:
        """
        Overview:
            Send learner's heartbeats to coordinator, will start as a thread in ``self.start_heartbeats_thread``
        """
        raise NotImplementedError

    @abstractproperty
    def hooks4call(self) -> list:
        """
        Returns:
            - hooks (:obj:`list`): the hooks which comm learner have, will be registered in learner as well.
        """
        raise NotImplementedError


class BaseCommSelfPlayLearner(object):

    def __init__(self):
        self._reset_ckpt_path = None

    def deal_with_reset_learner(self, ckpt_path: str) -> None:
        self._reset_ckpt_path = ckpt_path

    @property
    def reset_ckpt_path(self) -> str:
        ret = self._reset_ckpt_path
        self._reset_ckpt_path = None  # once reset_ckpt_path is used will it be set to None
        return ret
