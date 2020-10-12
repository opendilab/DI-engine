import threading
from abc import ABC, abstractmethod, abstractproperty

from nervex.utils import EasyTimer


class BaseCommLearner(ABC):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._learner_uid = None  # str(os.environ.get('SLURM_JOB_ID'))
        self._active_flag = False
        self._timer = EasyTimer()

    @abstractmethod
    def register_learner(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_agent(self, state_dict: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, batch_size: int) -> list:
        raise NotImplementedError

    @abstractmethod
    def send_train_info(self, train_info: dict) -> None:
        raise NotImplementedError

    def start_heartbeats_thread(self) -> None:
        check_send_learner_heartbeats_thread = threading.Thread(target=self._send_learner_heartbeats)
        check_send_learner_heartbeats_thread.daemon = True
        check_send_learner_heartbeats_thread.start()
        self._logger.info("Learner({}) send heartbeat thread start...".format(self._learner_uid))

    def init_service(self):
        self.register_learner()
        self._active_flag = True
        self.start_heartbeats_thread()

    def close_service(self):
        self._active_flag = False

    # ************************** thread *********************************
    @abstractmethod
    def _send_learner_heartbeats(self) -> None:
        raise NotImplementedError

    @abstractproperty
    def hooks4call(self) -> list:
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
